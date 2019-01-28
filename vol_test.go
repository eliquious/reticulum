package reticulum

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

func randArray(size int) []float64 {
	arr := make([]float64, size)
	for i := 0; i < len(arr); i++ {
		arr[i] = rand.Float64()
	}
	return arr
}

func TestWithInitialValue(t *testing.T) {
	type test struct {
		name string
		want float64
		fn   VolumeOptionFunc
	}
	tests := []test{
		{"Initializing with Zero", 0.0, WithInitialValue(0.0)},
		{"Initializing with 0.75", 0.75, WithInitialValue(0.75)},
	}

	for i := 0; i < 5; i++ {
		v := rand.Float64()
		tests = append(tests, test{name: fmt.Sprintf("Initializing with random[%d] %v", i, v), want: v, fn: WithInitialValue(v)})
	}

	opts := &VolumeOptions{}
	for _, tt := range tests {
		opts.HasInitialValue = false
		opts.InitialValue = -1.0
		opts.Zero = false
		t.Run(tt.name, func(t *testing.T) {
			tt.fn(opts)

			if !reflect.DeepEqual(opts.HasInitialValue, true) {
				t.Errorf("WithInitialValue() = %v, want %v", opts.HasInitialValue, true)
			}
			if !reflect.DeepEqual(opts.InitialValue, tt.want) {
				t.Errorf("WithInitialValue() = %v, want %v", opts.InitialValue, tt.want)
			}
			if !reflect.DeepEqual(opts.Zero, false) {
				t.Errorf("WithInitialValue() = %v, want %v", opts.Zero, false)
			}
		})
	}
}

func TestWithZeros(t *testing.T) {
	tests := []struct {
		name string
		fn   VolumeOptionFunc
	}{
		{"With Zeros", WithZeros()},
	}
	opts := &VolumeOptions{}
	for _, tt := range tests {
		opts.Zero = false
		opts.HasInitialValue = false
		t.Run(tt.name, func(t *testing.T) {
			tt.fn(opts)

			if !reflect.DeepEqual(opts.HasInitialValue, true) {
				t.Errorf("WithZeros() = %v, want %v", opts.HasInitialValue, true)
			}
			if !reflect.DeepEqual(opts.InitialValue, 0.0) {
				t.Errorf("WithInitialValue() = %v, want %v", opts.InitialValue, 0.0)
			}
			if !reflect.DeepEqual(opts.Zero, true) {
				t.Errorf("WithZeros() = %v, want %v", opts.Zero, true)
			}
		})
	}
}

func TestNewVolume(t *testing.T) {
	type args struct {
		sx       int
		sy       int
		depth    int
		optFuncs []VolumeOptionFunc
	}
	tests := []struct {
		name string
		args args
		want *Volume
	}{
		{"NewVolumeWithZeros", args{1, 1, 25, []VolumeOptionFunc{WithZeros()}}, NewVolume(1, 1, 25, WithZeros())},
		{"NewVolumeWithInitialValue", args{1, 1, 25, []VolumeOptionFunc{WithInitialValue(0.5)}}, NewVolume(1, 1, 25, WithInitialValue(0.5))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewVolume(tt.args.sx, tt.args.sy, tt.args.depth, tt.args.optFuncs...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewVolume() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVolume_Get(t *testing.T) {
	vol := NewVolume(1, 2, 6)
	for x := 0; x < vol.sx; x++ {
		for y := 0; y < vol.sy; y++ {
			for d := 0; d < vol.depth; d++ {
				want := vol.w[vol.getIndex(x, y, d)]
				if got := vol.Get(x, y, d); got != want {
					t.Errorf("Volume.Get() = %v, want %v", got, want)
				}
			}
		}
	}
}

func TestVolume_Set(t *testing.T) {
	vol := NewVolume(1, 2, 6, WithZeros())
	for x := 0; x < vol.sx; x++ {
		for y := 0; y < vol.sy; y++ {
			for d := 0; d < vol.depth; d++ {
				value := rand.Float64()
				vol.Set(x, y, d, value)
				want := vol.w[vol.getIndex(x, y, d)]

				if value != want {
					t.Errorf("Volume.Set() = %v, want %v", value, want)
				}
			}
		}
	}
}

func TestVolume_Add(t *testing.T) {
	type fields struct {
		sx    int
		sy    int
		depth int
		n     int
		w     []float64
		dw    []float64
	}
	type args struct {
		x   int
		y   int
		d   int
		val float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{"Zeros", fields{1, 2, 6, 12, make([]float64, 12), make([]float64, 12)}, args{1, 0, 5, 0.5}},
		{"Random", fields{1, 2, 6, 12, randArray(12), randArray(12)}, args{1, 0, 5, 0.1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &Volume{
				sx:    tt.fields.sx,
				sy:    tt.fields.sy,
				depth: tt.fields.depth,
				n:     tt.fields.n,
				w:     tt.fields.w,
				dw:    tt.fields.dw,
			}
			before := v.Get(tt.args.x, tt.args.y, tt.args.d)
			v.Add(tt.args.x, tt.args.y, tt.args.d, tt.args.val)

			value := v.Get(tt.args.x, tt.args.y, tt.args.d)
			want := before + tt.args.val
			if value != want {
				t.Errorf("Volume.Add() = %v, want %v", value, want)
			}
		})
	}
}

func TestVolume_getIndex(t *testing.T) {
	type args struct {
		x int
		y int
		d int
	}
	tests := []struct {
		name string
		args args
		vol  *Volume
		want int
	}{
		{"Zero XY", args{0, 0, 5}, NewVolume(1, 1, 20, WithZeros()), 5},
		{"Zero Y", args{1, 0, 5}, NewVolume(1, 1, 20, WithZeros()), 25},
		{"Zero X", args{0, 1, 5}, NewVolume(1, 1, 20, WithZeros()), 25},
		{"Zero D", args{1, 1, 0}, NewVolume(1, 20, 5, WithZeros()), 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := tt.vol
			if got := v.getIndex(tt.args.x, tt.args.y, tt.args.d); got != tt.want {
				t.Errorf("Volume.getIndex() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVolume_GetGrad(t *testing.T) {
	type fields struct {
		sx    int
		sy    int
		depth int
		n     int
		w     []float64
		dw    []float64
	}
	type args struct {
		x int
		y int
		d int
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   float64
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &Volume{
				sx:    tt.fields.sx,
				sy:    tt.fields.sy,
				depth: tt.fields.depth,
				n:     tt.fields.n,
				w:     tt.fields.w,
				dw:    tt.fields.dw,
			}
			if got := v.GetGrad(tt.args.x, tt.args.y, tt.args.d); got != tt.want {
				t.Errorf("Volume.GetGrad() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVolume_SetGrad(t *testing.T) {
	type fields struct {
		sx    int
		sy    int
		depth int
		n     int
		w     []float64
		dw    []float64
	}
	type args struct {
		x   int
		y   int
		d   int
		val float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &Volume{
				sx:    tt.fields.sx,
				sy:    tt.fields.sy,
				depth: tt.fields.depth,
				n:     tt.fields.n,
				w:     tt.fields.w,
				dw:    tt.fields.dw,
			}
			v.SetGrad(tt.args.x, tt.args.y, tt.args.d, tt.args.val)
		})
	}
}

func TestVolume_AddGrad(t *testing.T) {
	type fields struct {
		sx    int
		sy    int
		depth int
		n     int
		w     []float64
		dw    []float64
	}
	type args struct {
		x   int
		y   int
		d   int
		val float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &Volume{
				sx:    tt.fields.sx,
				sy:    tt.fields.sy,
				depth: tt.fields.depth,
				n:     tt.fields.n,
				w:     tt.fields.w,
				dw:    tt.fields.dw,
			}
			v.AddGrad(tt.args.x, tt.args.y, tt.args.d, tt.args.val)
		})
	}
}

func TestVolume_Clone(t *testing.T) {
	type fields struct {
		sx    int
		sy    int
		depth int
		n     int
		w     []float64
		dw    []float64
	}
	tests := []struct {
		name   string
		fields fields
		want   *Volume
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &Volume{
				sx:    tt.fields.sx,
				sy:    tt.fields.sy,
				depth: tt.fields.depth,
				n:     tt.fields.n,
				w:     tt.fields.w,
				dw:    tt.fields.dw,
			}
			if got := v.Clone(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Volume.Clone() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVolume_CloneAndZero(t *testing.T) {
	type fields struct {
		sx    int
		sy    int
		depth int
		n     int
		w     []float64
		dw    []float64
	}
	tests := []struct {
		name   string
		fields fields
		want   *Volume
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &Volume{
				sx:    tt.fields.sx,
				sy:    tt.fields.sy,
				depth: tt.fields.depth,
				n:     tt.fields.n,
				w:     tt.fields.w,
				dw:    tt.fields.dw,
			}
			if got := v.CloneAndZero(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Volume.CloneAndZero() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestVolume_AddFrom(t *testing.T) {
	type fields struct {
		sx    int
		sy    int
		depth int
		n     int
		w     []float64
		dw    []float64
	}
	type args struct {
		vol *Volume
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &Volume{
				sx:    tt.fields.sx,
				sy:    tt.fields.sy,
				depth: tt.fields.depth,
				n:     tt.fields.n,
				w:     tt.fields.w,
				dw:    tt.fields.dw,
			}
			v.AddFrom(tt.args.vol)
		})
	}
}

func TestVolume_AddFromScaled(t *testing.T) {
	type fields struct {
		sx    int
		sy    int
		depth int
		n     int
		w     []float64
		dw    []float64
	}
	type args struct {
		vol   *Volume
		scale float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := &Volume{
				sx:    tt.fields.sx,
				sy:    tt.fields.sy,
				depth: tt.fields.depth,
				n:     tt.fields.n,
				w:     tt.fields.w,
				dw:    tt.fields.dw,
			}
			v.AddFromScaled(tt.args.vol, tt.args.scale)
		})
	}
}

func TestVolume_SetConst(t *testing.T) {
	want := -1.0
	vol := NewVolume(1, 2, 6, WithInitialValue(-1.0))
	for x := 0; x < vol.sx; x++ {
		for y := 0; y < vol.sy; y++ {
			for d := 0; d < vol.depth; d++ {
				if got := vol.Get(x, y, d); got != want {
					t.Errorf("Volume.Get() = %v, want %v", got, want)
				}
			}
		}
	}

	want = 5.0
	vol.SetConst(want)
	for x := 0; x < vol.sx; x++ {
		for y := 0; y < vol.sy; y++ {
			for d := 0; d < vol.depth; d++ {
				if got := vol.Get(x, y, d); got != want {
					t.Errorf("Volume.Get() = %v, want %v", got, want)
				}
			}
		}
	}
}
