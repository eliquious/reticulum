package layers

import (
	"fmt"
	"math"

	"github.com/eliquious/reticulum/volume"
)

// NewSoftmaxLayer creates a new softmax layer.
// This is a classifier, with N discrete classes from 0 to N-1. It gets a stream
// of N incoming numbers and computes the softmax function (exponentiate and
// normalize to sum to 1 as probabilities should)
func NewSoftmaxLayer(def LayerDef) Layer {
	if def.Type != SoftMax {
		panic(fmt.Errorf("Invalid layer type: %s != softmax", def.Type))
	}

	n := def.Input.Size()
	return &softmaxLayer{def.Input, volume.Dimensions{1, 1, n}, nil, nil, []float64{}}
}

type softmaxLayer struct {
	inDim  volume.Dimensions
	outDim volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume

	es []float64
}

func (l *softmaxLayer) Type() LayerType {
	return SoftMax
}

func (l *softmaxLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.inVol = vol

	n := l.outDim.Z
	volA := volume.NewVolume(l.outDim, volume.WithZeros())

	// compute max activation
	as := vol.Weights()
	aMax := as[0]
	for i := 0; i < n; i++ {
		if as[i] > aMax {
			aMax = as[i]
		}
	}

	// compute exponentials (carefully to not blow up)
	es := make([]float64, n, n)
	esum := 0.0
	for i := 0; i < n; i++ {
		e := math.Exp(as[i] - aMax)
		esum += e
		es[i] = e
	}

	// normalize and output to sum to one
	for i := 0; i < n; i++ {
		es[i] /= esum
		volA.SetByIndex(i, es[i])
	}

	// save these for backprop
	l.es = es
	l.outVol = volA
	return l.outVol
}

func (l *softmaxLayer) Loss(index int) float64 {

	// compute and accumulate gradient wrt weights and bias of this layer
	// zero out the gradient of input Vol
	l.inVol.ZeroGrad()

	n := l.outDim.Z
	for i := 0; i < n; i++ {
		indicator := 0.0
		if i == index {
			indicator = 1.0
		}

		l.inVol.SetGradByIndex(i, -(indicator - l.es[i]))
	}

	// loss is the class negative log likelihood
	return -math.Log(l.es[index])
}

func (l *softmaxLayer) Backward() {
}

func (l *softmaxLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
