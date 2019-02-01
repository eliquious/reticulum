package layers

import (
	"fmt"

	"github.com/eliquious/reticulum/volume"
)

// NewSVMLayer creates a new SVM layer.
func NewSVMLayer(def LayerDef) Layer {
	if def.Type != SVM {
		panic(fmt.Errorf("Invalid layer type: %s != svm", def.Type))
	}

	n := def.Input.Size()
	return &svmLayer{def.Input, volume.Dimensions{1, 1, n}, nil, nil}
}

type svmLayer struct {
	inDim  volume.Dimensions
	outDim volume.Dimensions

	inVol  *volume.Volume
	outVol *volume.Volume
}

func (l *svmLayer) Type() LayerType {
	return SVM
}

func (l *svmLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.inVol = vol
	l.outVol = vol
	return vol
}

func (l *svmLayer) Loss(index int) float64 {
	if index < 0 || index >= l.outDim.Size() {
		panic(fmt.Errorf("Invalid dimension index: %d", index))
	}

	// compute and accumulate gradient wrt weights and bias of this layer
	// zero out the gradient of input Vol
	l.inVol.ZeroGrad()

	// score of ground truth
	yScore := l.inVol.GetByIndex(index)

	// we're using structured loss here, which means that the score
	// of the ground truth should be higher than the score of any other
	// class, by a margin

	var loss float64
	margin := 1.0
	for i := 0; i < l.outVol.Size(); i++ {
		if index == i {
			continue
		}

		yDiff := -yScore + l.inVol.GetByIndex(i) + margin
		if yDiff > 0 {
			// violating dimension, apply loss
			l.inVol.AddGradByIndex(i, 1.0)
			l.inVol.AddGradByIndex(index, -1.0)
			loss += yDiff
		}
	}
	return loss
}

func (l *svmLayer) Backward() {
	panic(fmt.Errorf("Unsupported operation"))
}

func (l *svmLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
