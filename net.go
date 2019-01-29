package reticulum

import "github.com/eliquious/reticulum/volume"
import "github.com/eliquious/reticulum/layers"

// Network is the neural network interface.
type Network interface {
	Forward(vol volume.Volume, training bool)
	Backward(index int)
	GetCostLoss(vol volume.Volume, index int)
	GetPrediction() int
	GetResponse() []layers.LayerResponse
}

type network struct {
	layers []layers.Layer
}

func NewNetwork(defs []layers.Layer) (Network, error) {
	if len(defs) <= 2 {
		return nil, errors.New("At least one input and one loss layer are required.")
	} else if defs[0].Type != layers.Input {
		return nil, errors.New("First layer must be the input layer, to declare size of inputs.")
	}

	// TODO: Complete
	return nil, nil
}
