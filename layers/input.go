package layers

import "github.com/eliquious/reticulum/volume"

// NewInputLayer creates a new input layer.
func NewInputLayer(depth int) Layer {
	return &InputLayer{1, 1, depth, nil, nil}
}

type InputLayer struct {
	outSx    int
	outSy    int
	outDepth int

	inVol  *volume.Volume
	outVol *volume.Volume
}

func (il *InputLayer) Type() LayerType {
	return Input
}

func (il *InputLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	il.inVol = vol
	il.outVol = vol
	return il.outVol
}

func (il *InputLayer) Backward() {}

func (il *InputLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
