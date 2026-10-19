package cli

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/TensorRails/warmctl/v2/internal/host"
	"github.com/TensorRails/warmctl/v2/internal/profile"
	"github.com/TensorRails/warmctl/v2/internal/snapshot"
)

// runHardware prints the hardware table, the shipped rows and those of a
// hardware.yaml in each --dir, and marks the rows of this host's GPUs.
func runHardware(ctx context.Context, args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("hardware", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var dirs list
	fs.Var(&dirs, "dir", "extra directory with a hardware.yaml (repeatable)")
	pos, err := positionals(fs, args)
	if err != nil {
		return err
	}
	if len(pos) > 0 {
		return errors.New("usage: warmctl hardware [--dir DIR]")
	}
	table, err := profileLoader(dirs).Hardware()
	if err != nil {
		return err
	}
	// Without NVML (no driver, macOS) the table is listed unmarked.
	gpus, _ := host.Inventory(ctx)
	models := hostModels(table, gpus)
	here := map[string]int{}
	for _, m := range models {
		if m.err == nil {
			here[m.row.ID] = m.count
		}
	}
	if err := printHardware(stdout, table.Rows, here); err != nil {
		return err
	}
	for _, m := range models {
		if m.err != nil {
			fmt.Fprintf(stdout, "this host: %d x %s, no row; add one to a hardware.yaml in a --dir\n", m.count, m.name)
		}
	}
	return nil
}

// printHardware writes the rows as a table, with a THIS HOST column of the
// GPU counts in here, by row id, when there are any.
func printHardware(w io.Writer, rows []profile.Hardware, here map[string]int) error {
	var buf bytes.Buffer
	tw := tabwriter.NewWriter(&buf, 0, 4, 2, ' ', 0)
	fmt.Fprint(tw, "ID\tNVIDIA-SMI NAME\tMEMORY\tARCH\tNVLINK\tTOPOLOGY")
	if len(here) > 0 {
		fmt.Fprint(tw, "\tTHIS HOST")
	}
	fmt.Fprintln(tw)
	for _, r := range rows {
		nvlink := "no"
		if r.NVLink {
			nvlink = "yes"
		}
		fmt.Fprintf(tw, "%s\t%s\t%d MiB\t%s\t%s\t%s", r.ID, r.NvidiaSMIName, r.MemoryMiB, r.Arch, nvlink, r.TopologyClass)
		switch n := here[r.ID]; {
		case n == 1:
			fmt.Fprint(tw, "\t1 GPU")
		case n > 1:
			fmt.Fprintf(tw, "\t%d GPUs", n)
		case len(here) > 0:
			fmt.Fprint(tw, "\t")
		}
		fmt.Fprintln(tw)
	}
	if err := tw.Flush(); err != nil {
		return err
	}
	// The padding before an empty THIS HOST cell would end the line.
	for _, l := range strings.Split(strings.TrimSuffix(buf.String(), "\n"), "\n") {
		fmt.Fprintln(w, strings.TrimRight(l, " "))
	}
	return nil
}

// gpuModel is one GPU model of the host and its hardware row.
type gpuModel struct {
	name  string
	count int
	row   profile.Hardware
	err   error
}

// hostModels groups the host's GPUs by model, in inventory order, and
// matches each model to its row as build does.
func hostModels(table profile.HardwareTable, gpus []host.GPU) []gpuModel {
	var names []string
	byName := map[string][]host.GPU{}
	for _, g := range gpus {
		if len(byName[g.Name]) == 0 {
			names = append(names, g.Name)
		}
		byName[g.Name] = append(byName[g.Name], g)
	}
	out := make([]gpuModel, len(names))
	for i, name := range names {
		group := byName[name]
		row, err := snapshot.Hardware(table, group, len(group))
		out[i] = gpuModel{name: name, count: len(group), row: row, err: err}
	}
	return out
}
