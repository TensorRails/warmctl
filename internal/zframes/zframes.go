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
	var ends []int64
	var end int64
	batch := make([][]byte, workers)
	for {
		// A batch of frames is read in order, compressed at once, written in order.
		n := 0
		for ; n < workers; n++ {
			if batch[n] == nil {
				batch[n] = make([]byte, Size)
			}
			got, err := io.ReadFull(in, batch[n][:Size])
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, 0, errors.Join(err, out.Close())
			}
			batch[n] = batch[n][:got]
			if got < Size {
				n++
				break
			}
		}
		if n == 0 {
			break
		}
		if split == 0 {
			split = 1
			if len(enc.EncodeAll(planes(batch[0], 2), nil)) < len(enc.EncodeAll(batch[0], nil)) {
				split = 2
			}
		}
		frames := make([][]byte, n)
		var wg sync.WaitGroup
		for i := range n {
			wg.Add(1)
			go func() {
				defer wg.Done()
				frames[i] = enc.EncodeAll(planes(batch[i], split), nil)
			}()
		}
		wg.Wait()
		for _, f := range frames {
			if _, err := out.Write(f); err != nil {
				return nil, 0, errors.Join(err, out.Close())
			}
			end += int64(len(f))
			ends = append(ends, end)
		}
		if err := ctx.Err(); err != nil {
			return nil, 0, errors.Join(err, out.Close())
		}
		if len(batch[n-1]) < Size {
			break
		}
	}
	if split == 0 {
		split = 1
	}
	return ends, split, out.Close()
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
