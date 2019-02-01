package layers

import (
	"fmt"

	"github.com/eliquious/reticulum/volume"
)

// NewRegressionLayer creates a new regression layer.
func NewRegressionLayer(def LayerDef) Layer {
	if def.Type != Regression {
		panic(fmt.Errorf("Invalid layer type: %s != regression", def.Type))
	}

	n := def.Input.Size()
	return &regressionLayer{def.Input, volume.Dimensions{1, 1, n}, nil, nil}
}

type regressionLayer struct {
	inDim  volume.Dimensions
	outDim volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume
}

func (l *regressionLayer) Type() LayerType {
	return Regression
}

func (l *regressionLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.inVol = vol
	l.outVol = vol
	return vol
}

func (l *regressionLayer) MultiDimensionalLoss(y []float64) float64 {
	if len(y) != l.outDim.Size() {
		panic(fmt.Errorf("Invalid input length: %d != %d", len(y), l.outDim.Size()))
	}

	// compute and accumulate gradient wrt weights and bias of this layer
	// zero out the gradient of input Vol
	l.inVol.ZeroGrad()

	var loss float64
	for i := 0; i < l.outDim.Size(); i++ {
		dY := l.inVol.GetByIndex(i) - y[i]
		l.inVol.SetGradByIndex(i, dY)
		loss += 0.5 * dY * dY
	}

	return loss
}

func (l *regressionLayer) DimensionalLoss(index int, value float64) float64 {
	if index < 0 || index >= l.outDim.Size() {
		panic(fmt.Errorf("Invalid dimension index: %d", index))
	}

	// compute and accumulate gradient wrt weights and bias of this layer
	// zero out the gradient of input Vol
	l.inVol.ZeroGrad()

	// assume it is a struct with entries .dim and .val
	// and we pass gradient only along dimension dim to be equal to val
	var loss float64
	dY := l.inVol.GetByIndex(index) - value
	l.inVol.SetGradByIndex(index, dY)
	loss += 0.5 * dY * dY
	return loss
}

func (l *regressionLayer) Backward() {
	panic(fmt.Errorf("Unsupported operation"))
}

func (l *regressionLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
