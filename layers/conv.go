package layers

import (
	"fmt"
	"math"

	"github.com/eliquious/reticulum/volume"
)

// WithConvStride sets the stride for the conv layer
func WithConvStride(stride int) LayerOptionFunc {
	return func(lc LayerConfig) error {
		conf, ok := lc.(*convLayerConfig)
		if !ok {
			return fmt.Errorf("Invalid LayerConfig for ConfLayer")
		}
		conf.Stride = stride
		return nil
	}
}

// WithConvPadding sets the padding for the conv layer
func WithConvPadding(pad int) LayerOptionFunc {
	return func(lc LayerConfig) error {
		conf, ok := lc.(*convLayerConfig)
		if !ok {
			return fmt.Errorf("Invalid LayerConfig for ConfLayer")
		}
		conf.Padding = pad
		return nil
	}
}

// WithConvDecay sets the L1 & L2 decay for the conv layer
func WithConvDecay(l1 float64, l2 float64) LayerOptionFunc {
	return func(lc LayerConfig) error {
		conf, ok := lc.(*convLayerConfig)
		if !ok {
			return fmt.Errorf("Invalid LayerConfig for ConfLayer")
		}
		conf.L1DecayMult = l1
		conf.L2DecayMult = l2
		return nil
	}
}

// WithConvBias sets the preferred bias for the conv layer
func WithConvBias(bias float64) LayerOptionFunc {
	return func(lc LayerConfig) error {
		conf, ok := lc.(*convLayerConfig)
		if !ok {
			return fmt.Errorf("Invalid LayerConfig for ConfLayer")
		}
		conf.PreferredBias = bias
		return nil
	}
}

// WithConvSx sets the sx for the conv layer
func WithConvSx(sx int) LayerOptionFunc {
	return func(lc LayerConfig) error {
		conf, ok := lc.(*convLayerConfig)
		if !ok {
			return fmt.Errorf("Invalid LayerConfig for ConfLayer")
		}
		conf.Sx = sx
		return nil
	}
}

// WithConvSy sets the sy for the conv layer
func WithConvSy(sy int) LayerOptionFunc {
	return func(lc LayerConfig) error {
		conf, ok := lc.(*convLayerConfig)
		if !ok {
			return fmt.Errorf("Invalid LayerConfig for ConfLayer")
		}
		conf.Sy = sy
		return nil
	}
}

// NewConvLayerConfig creates a new ConvLayer config with the given options.
func NewConvLayerConfig(filters int, opts ...LayerOptionFunc) LayerConfig {
	if filters <= 0 {
		panic("Filter count must be greater than 0")
	}

	conf := &convLayerConfig{
		FilterCount:   filters,
		Stride:        1,
		Padding:       0,
		L1DecayMult:   0.0,
		L2DecayMult:   1.0,
		PreferredBias: 0.0,
	}
	for i := 0; i < len(opts); i++ {
		err := opts[i](conf)
		if err != nil {
			panic(err)
		}
	}
	return conf
}

type convLayerConfig struct {
	FilterCount   int
	Sx            int
	Sy            int
	Stride        int
	Padding       int
	L1DecayMult   float64
	L2DecayMult   float64
	PreferredBias float64
}

// NewConvLayer creates a new convoluted layer.
func NewConvLayer(def LayerDef) Layer {

	// Validate input
	if def.Type != Conv {
		panic(fmt.Errorf("Invalid layer type: %s != conv", def.Type))
	} else if def.Output.Z == 0 {
		panic(fmt.Errorf("Output depth cannot be 0 for conv layer"))
	} else if def.LayerConfig == nil {
		panic(fmt.Errorf("Config cannot be nil for conv layer"))
	}

	// Get config
	conf, ok := def.LayerConfig.(*convLayerConfig)
	if !ok {
		panic("Invalid LayerConfig for ConvLayer")
	}

	// Set Sy
	if conf.Sy <= 0 {
		conf.Sy = conf.Sx
	}

	// Output dimensions
	outDepth := conf.FilterCount
	outSx := math.Floor((float64(def.Input.X)+float64(conf.Padding)*2.0-float64(conf.Sx))/float64(conf.Stride) + 1)
	outSy := math.Floor((float64(def.Input.Y)+float64(conf.Padding)*2.0-float64(conf.Sy))/float64(conf.Stride) + 1)
	outDim := volume.Dimensions{int(outSx), int(outSy), outDepth}

	bias := conf.PreferredBias
	var filters []*volume.Volume
	for i := 0; i < outDepth; i++ {
		filters = append(filters, volume.NewVolume(volume.Dimensions{conf.Sx, conf.Sy, def.Input.Z}))
	}

	biases := volume.NewVolume(volume.Dimensions{1, 1, outDepth}, volume.WithInitialValue(bias))
	return &convLayer{conf, def.Input, outDim, nil, nil, filters, biases}
}

type convLayer struct {
	conf   *convLayerConfig
	input  volume.Dimensions
	output volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume

	filters []*volume.Volume
	biases  *volume.Volume
}

func (*convLayer) Type() LayerType {
	return Conv
}

func (l *convLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.inVol = vol
	A := volume.NewVolume(l.output, volume.WithZeros())

	vDim := vol.Dimensions()
	vsx, vsy, stride := vDim.X, vDim.Y, l.conf.Stride
	for d := 0; d < l.output.Z; d++ {
		f := l.filters[d]
		y := -l.conf.Padding
		for ay := 0; ay < l.output.Y; ay++ {
			y += stride
			x := -l.conf.Padding
			for ax := 0; ax < l.output.X; ax++ {
				x += stride

				var a float64
				fDim := f.Dimensions()
				for fy := 0; fy < fDim.Y; fy++ {
					oy := y + fy
					for fx := 0; fx < fDim.X; fx++ {
						ox := x + fx
						if oy >= 0 && oy < vsy && ox >= 0 && ox < vsx {
							for fz := 0; fz < fDim.Z; fz++ {
								a1 := f.GetByIndex(((fDim.X*fy)+fx)*fDim.Z + fz)
								a2 := vol.GetByIndex(((vsx*oy)+ox)*vDim.Z + fz)
								a += a1 * a2
							}
						}
					}
				}
				a += l.biases.GetByIndex(d)
				A.Set(ax, ay, d, a)
			}
		}
	}

	l.outVol = A
	return l.outVol
}

func (l *convLayer) Backward() {
	l.inVol.ZeroGrad()

	vDim := l.inVol.Dimensions()
	vsx, vsy, stride := vDim.X, vDim.Y, l.conf.Stride

	for d := 0; d < l.output.Z; d++ {
		f := l.filters[d]
		y := -l.conf.Padding

		fDim := f.Dimensions()
		for ay := 0; ay < l.output.Y; ay++ {
			y += stride
			x := -l.conf.Padding
			for ax := 0; ax < l.output.X; ax++ {
				x += stride
				chainGrad := l.outVol.GetGrad(ax, ay, d)
				for fy := 0; fy < fDim.Y; fy++ {
					oy := y + fy
					for fx := 0; fx < fDim.X; fx++ {
						ox := x + fx
						if oy >= 0 && oy < vsy && ox >= 0 && ox < vsx {
							for fz := 0; fz < fDim.Z; fz++ {
								ix1 := ((vsy*oy)+ox)*vDim.Z + fz
								ix2 := ((fDim.X*fy)+fx)*fDim.Z + fz
								f.SetGradByIndex(ix2, l.inVol.GetByIndex(ix1)*chainGrad)
								l.inVol.SetGradByIndex(ix1, f.GetByIndex(ix2)*chainGrad)
							}
						}
					}
				}
				l.biases.AddGradByIndex(d, chainGrad)
			}
		}
	}
}

func (l *convLayer) GetResponse() []LayerResponse {
	var resp []LayerResponse
	for i := 0; i < l.output.Z; i++ {
		resp = append(resp, LayerResponse{
			Weights:    l.filters[i].Weights(),
			Gradients:  l.filters[i].Gradients(),
			L1DecayMul: l.conf.L1DecayMult,
			L2DecayMul: l.conf.L2DecayMult,
		})
	}
	resp = append(resp, LayerResponse{
		Weights:    l.biases.Weights(),
		Gradients:  l.biases.Gradients(),
		L1DecayMul: 0.0,
		L2DecayMul: 0.0,
	})
	return resp
}
