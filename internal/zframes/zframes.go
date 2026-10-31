// Package zframes stores a file as independent zstd frames, each of Size
// bytes of it but the last, so that a reader fetches and decodes the frames
// in parallel: the form a snapshot's process images and weights take on
// network storage, where every byte read costs and one decoding stream
// would not keep up with fast storage.
//
// A frame may hold its bytes split into planes: with split 2, the bytes at
// even offsets and then those at odd ones. BF16 weights keep their sign and
// exponent in the high byte, which compresses to about a third, and their
// mantissa in the low one, which does not compress at all: split, a 16-bit
// tensor takes 0.69 of its size, where zstd alone gets 0.78.
package zframes

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/klauspost/compress/zstd"
)

// Size is how many bytes of the file one frame holds.
const Size = 32 << 20

// Encode writes src to dst as frames, workers at a time, and returns each
// frame's end offset in dst and the split it used: split, or with 0 the one
// of 1 and 2 that compresses the first frame best.
func Encode(ctx context.Context, src, dst string, workers, split int) ([]int64, int, error) {
	in, err := os.Open(src)
	if err != nil {
		return nil, 0, err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return nil, 0, err
	}
	enc, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedFastest), zstd.WithEncoderConcurrency(1),
		zstd.WithEncoderCRC(true))
	if err != nil {
		return nil, 0, errors.Join(err, out.Close())
	}
	e := &encoder{enc: enc, batch: make([][]byte, workers), split: split}
	ends, err := e.copy(ctx, in, out)
	if err != nil {
		return nil, 0, errors.Join(err, out.Close())
	}
	return ends, max(e.split, 1), out.Close()
}

// encoder compresses a batch of frames at a time: read in order, compressed
// at once, written in order.
type encoder struct {
	enc   *zstd.Encoder
	batch [][]byte
	split int
}

func (e *encoder) copy(ctx context.Context, in io.Reader, out io.Writer) ([]int64, error) {
	var ends []int64
	var end int64
	for {
		n, last, err := e.read(in)
		if err != nil || n == 0 {
			return ends, err
		}
		e.choose(e.batch[0])
		for _, f := range e.compress(n) {
			if _, err := out.Write(f); err != nil {
				return nil, err
			}
			end += int64(len(f))
			ends = append(ends, end)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if last {
			return ends, nil
		}
	}
}

// read fills the batch with up to len(batch) frames and says whether in
// ended.
func (e *encoder) read(in io.Reader) (int, bool, error) {
	for n := range e.batch {
		if e.batch[n] == nil {
			e.batch[n] = make([]byte, Size)
		}
		got, err := io.ReadFull(in, e.batch[n][:Size])
		e.batch[n] = e.batch[n][:got]
		switch {
		case errors.Is(err, io.EOF):
			return n, true, nil
		case errors.Is(err, io.ErrUnexpectedEOF):
			return n + 1, true, nil
		case err != nil:
			return 0, false, err
		}
	}
	return len(e.batch), false, nil
}

// choose settles an unset split on the first frame.
func (e *encoder) choose(first []byte) {
	if e.split != 0 {
		return
	}
	e.split = 1
	if len(e.enc.EncodeAll(planes(first, 2), nil)) < len(e.enc.EncodeAll(first, nil)) {
		e.split = 2
	}
}

// compress encodes the batch's first n frames at once.
func (e *encoder) compress(n int) [][]byte {
	frames := make([][]byte, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			frames[i] = e.enc.EncodeAll(planes(e.batch[i], e.split), nil)
		}()
	}
	wg.Wait()
	return frames
}

// Frame is where frame i of a file of size bytes lies, in the stored file
// (ends from Encode) and in the file itself.
func Frame(ends []int64, size int64, i int) (stored, storedSize, offset, n int64) {
	if i > 0 {
		stored = ends[i-1]
	}
	offset = int64(i) * Size
	return stored, ends[i] - stored, offset, min(Size, size-offset)
}

// Decoder decodes one frame at a time; one per goroutine.
type Decoder struct {
	d      *zstd.Decoder
	planes []byte
}

// NewDecoder makes a Decoder.
func NewDecoder() (*Decoder, error) {
	d, err := zstd.NewReader(nil, zstd.WithDecoderConcurrency(1), zstd.WithDecoderMaxMemory(2*Size))
	if err != nil {
		return nil, err
	}
	return &Decoder{d: d}, nil
}

// Decode decodes frame, stored with split, and checks it holds want bytes.
// The bytes are in out, or the decoder's own buffer: valid until the next
// Decode.
func (d *Decoder) Decode(frame, out []byte, want int64, split int) ([]byte, error) {
	var err error
	d.planes, err = d.d.DecodeAll(frame, d.planes[:0])
	if err != nil {
		return nil, err
	}
	if int64(len(d.planes)) != want {
		return nil, fmt.Errorf("zframes: frame holds %d bytes, want %d", len(d.planes), want)
	}
	if split <= 1 {
		return d.planes, nil
	}
	return interleave(out, d.planes, split), nil
}

// planes is b with its bytes in split planes: those at offsets 0, split,
// 2*split... first; a tail shorter than split stays at the end.
func planes(b []byte, split int) []byte {
	if split <= 1 {
		return b
	}
	n := len(b) / split
	out := make([]byte, len(b))
	for k := range split {
		p := out[k*n : (k+1)*n]
		for i := range p {
			p[i] = b[i*split+k]
		}
	}
	copy(out[n*split:], b[n*split:])
	return out
}

// interleave undoes planes into out[:0].
func interleave(out, p []byte, split int) []byte {
	if cap(out) < len(p) {
		out = make([]byte, len(p))
	}
	out = out[:len(p)]
	n := len(p) / split
	for k := range split {
		src := p[k*n : (k+1)*n]
		for i, c := range src {
			out[i*split+k] = c
		}
	}
	copy(out[n*split:], p[n*split:])
	return out
}

// Close frees the decoder.
func (d *Decoder) Close() { d.d.Close() }
