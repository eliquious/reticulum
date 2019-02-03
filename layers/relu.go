package layers

import "github.com/eliquious/reticulum/volume"
import "fmt"

// NewReluLayer creates a new ReLU (rectified linear unit) layer.
func NewReluLayer(def LayerDef) Layer {
	if def.Type != ReLU {
		panic(fmt.Errorf("Invalid layer type: %s != relu", def.Type))
	} else if def.Output.Z == 0 {
		panic(fmt.Errorf("Output depth cannot be 0 for relu layer"))
	}
	return &reluLayer{def.Output, nil, nil}
}

type reluLayer struct {
	output volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume
}

func (il *reluLayer) Type() LayerType {
	return ReLU
}

func (il *reluLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	il.inVol = vol
	v2 := vol.Clone()

	// Rectify to zero
	n := vol.Size()
	for i := 0; i < n; i++ {
		if v2.GetByIndex(i) < 0 {
			v2.SetByIndex(i, 0)
		}
	}

	il.outVol = v2
	return il.outVol
}

func (il *reluLayer) Backward() {
	n := l.inVol.Size()
	l.inVol.ZeroGrad()

	// Set the gradient of the input if the output is below threshold (0)
	for i := 0; i < n; i++ {
		// Threshold
		if l.outVol.GetByIndex(i) <= 0 {
			l.inVol.SerGradByIndex(i, 0)
		} else {
			l.inVol.SetGradByIndex(i, l.outVol.GetGradByIndex(i))
		}
	}
}

func (il *reluLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
