// Package zframes stores a file as independent zstd frames, each of Size
// bytes of it but the last, so that a reader fetches and decodes the frames
// in parallel: the form a snapshot's process images take on network
// storage, where every byte read costs and one decoding stream would not
// keep up with fast storage.
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
// frame's end offset in dst.
func Encode(ctx context.Context, src, dst string, workers int) ([]int64, error) {
	in, err := os.Open(src)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return nil, err
	}
	enc, err := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedFastest), zstd.WithEncoderConcurrency(1),
		zstd.WithEncoderCRC(true))
	if err != nil {
		return nil, errors.Join(err, out.Close())
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
				return nil, errors.Join(err, out.Close())
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
		frames := make([][]byte, n)
		var wg sync.WaitGroup
		for i := range n {
			wg.Add(1)
			go func() {
				defer wg.Done()
				frames[i] = enc.EncodeAll(batch[i], nil)
			}()
		}
		wg.Wait()
		for _, f := range frames {
			if _, err := out.Write(f); err != nil {
				return nil, errors.Join(err, out.Close())
			}
			end += int64(len(f))
			ends = append(ends, end)
		}
		if err := ctx.Err(); err != nil {
			return nil, errors.Join(err, out.Close())
		}
		if len(batch[n-1]) < Size {
			break
		}
	}
	return ends, out.Close()
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
type Decoder struct{ d *zstd.Decoder }

// NewDecoder makes a Decoder.
func NewDecoder() (*Decoder, error) {
	d, err := zstd.NewReader(nil, zstd.WithDecoderConcurrency(1), zstd.WithDecoderMaxMemory(2*Size))
	if err != nil {
		return nil, err
	}
	return &Decoder{d}, nil
}

// Decode decodes frame into out[:0] and checks it holds want bytes.
func (d *Decoder) Decode(frame, out []byte, want int64) ([]byte, error) {
	out, err := d.d.DecodeAll(frame, out[:0])
	if err != nil {
		return nil, err
	}
	if int64(len(out)) != want {
		return nil, fmt.Errorf("zframes: frame holds %d bytes, want %d", len(out), want)
	}
	return out, nil
}

// Close frees the decoder.
func (d *Decoder) Close() { d.d.Close() }
