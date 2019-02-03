package layers

import (
	"fmt"
	"math"

	"github.com/eliquious/reticulum/volume"
)

// NewSigmoidLayer creates a new Sigmoid layer.
func NewSigmoidLayer(def LayerDef) Layer {
	if def.Type != Sigmoid {
		panic(fmt.Errorf("Invalid layer type: %s != sigmoid", def.Type))
	} else if def.Output.Z == 0 {
		panic(fmt.Errorf("Output depth cannot be 0 for sigmoid layer"))
	}
	return &sigmoidLayer{def.Output, nil, nil}
}

type sigmoidLayer struct {
	output volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume
}

func (il *sigmoidLayer) Type() LayerType {
	return Sigmoid
}

func (il *sigmoidLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	il.inVol = vol
	v2 := vol.CloneAndZero()

	// Rectify to zero
	n := vol.Size()
	for i := 0; i < n; i++ {
		v2.SetByIndex(i, 1.0/(1.0+math.Exp(-vol.GetByIndex(i))))
	}

	il.outVol = v2
	return il.outVol
}

func (il *sigmoidLayer) Backward() {
	n := l.inVol.Size()
	l.inVol.ZeroGrad()

	for i := 0; i < n; i++ {
		v2wi := l.outVol.GetByIndex(i)
		l.inVol.SetGradByIndex(i, v2wi*(1-v2wi)*l.outVol.GetGradByIndex(i))
	}
}

func (il *sigmoidLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
