package layers

import "github.com/eliquious/reticulum/volume"

// NewDropoutLayer creates a new dropout layer.
func NewDropoutLayer(sx, sy, depth int) Layer {
	n := sx * sy * depth
	return &dropoutLayer{sx, sy, depth, 0.5, make([]float64, n, n), nil, nil}
}

// NewDropoutLayer creates a new dropout layer.
func NewDropoutLayerWithProb(sx, sy, depth int, prob float64) Layer {
	n := sx * sy * depth
	return &dropoutLayer{sx, sy, depth, prob, make([]bool, n, n), nil, nil}
}

type DropoutLayer struct {
	outSx    int
	outSy    int
	outDepth int

	DropoutProb float64
	dropped     []bool

	inVol  *volume.Volume
	outVol *volume.Volume
}

func (l *DropoutLayer) Type() LayerType {
	return Dropout
}

func (l *DropoutLayer) Forward(vol *volume.Volume, training bool) *volume.Volume {
	l.in_act = vol
	vol2 := vol.Clone()
	n := vol.Size()

	if training {
		// Perform dropout based on probabilty
		for i := 0; i < n; i++ {
			if rand.Float64() < l.DropoutProb {
				vol2.SetByIndex(i, 0.0)
				l.dropped[i] = true
			} else {
				l.dropped[i] = false
			}
		}
	} else {
		// scale the activations during prediction
		for i := 0; i < n; i++ {
			vol2.MultByIndex(i, l.DropoutProb)
		}
	}

	l.outVol = vol2
	return l.outVol
}

func (l *DropoutLayer) Backward() {

	// Need to set the gradients to zero
	l.inVol.ZeroGrad()
	chainGrad := l.outVol
	n := vol.Size()

	// Apply dropouts to input volume
	for i := 0; i < n; i++ {
		if !l.dropped[i] {

			// copy over the gradient
			l.inVol.SetGradByIndex(i, chainGrad.GetGradByIndex(i))
		}
	}
}

func (l *DropoutLayer) GetResponse() []LayerResponse {
	return []LayerResponse{}
}
