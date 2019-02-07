package reticulum

import (
	"errors"

	layers "github.com/eliquious/reticulum/layers"
	volume "github.com/eliquious/reticulum/volume"
)

const (
	// DefaultDropout is the default dropout rate of 0.5 or 50%. Everything less than the dropout rate will be dropped.
	DefaultDropout float64 = 0.5
)

// Network is the neural network interface.
type Network interface {
	Forward(vol volume.Volume, training bool)
	Backward(index int)
	GetCostLoss(vol volume.Volume, index int)
	GetPrediction() int
	GetResponse() []layers.LayerResponse

	MultiDimensionalLoss(losses []float64) float64
	DimensionalLoss(index int, value float64) float64
}

// NewNetwork creates a new network from the layer definitions
func NewNetwork(defs []layers.LayerDef) (Network, error) {
	if len(defs) <= 2 {
		return nil, errors.New("at least one input and one loss layer are required")
	} else if defs[0].Type != layers.Input {
		return nil, errors.New("first layer must be the input layer, to declare size of inputs")
	}

	// Add activation layers
	defs = layers.ActivateLayers(defs)

	var newLayers []layers.Layer
	for i, def := range defs {
		if i > 0 {
			prev := defs[i-1]
			def.Input = prev.Output
		}

		switch def.Type {
		case layers.FullyConnected:
			newLayers = append(newLayers, layers.NewFullyConnectedLayer(def))
		case layers.Dropout:
			newLayers = append(newLayers, layers.NewDropoutLayer(def))
		case layers.Input:
			newLayers = append(newLayers, layers.NewInputLayer(def))
		case layers.SoftMax:
			newLayers = append(newLayers, layers.NewSoftmaxLayer(def))
		case layers.Regression:
			newLayers = append(newLayers, layers.NewRegressionLayer(def))
		case layers.Conv:
			newLayers = append(newLayers, layers.NewConvLayer(def))
		case layers.Pool:
			newLayers = append(newLayers, layers.NewPoolLayer(def))
		case layers.ReLU:
			newLayers = append(newLayers, layers.NewReluLayer(def))
		case layers.Sigmoid:
			newLayers = append(newLayers, layers.NewSigmoidLayer(def))
		case layers.Tanh:
			newLayers = append(newLayers, layers.NewTanhLayer(def))
		case layers.Maxout:
			newLayers = append(newLayers, layers.NewMaxoutLayer(def))
		case layers.SVM:
			newLayers = append(newLayers, layers.NewSVMLayer(def))
		// case layers.LocalResponseNorm:
		default:
			return nil, errors.New("unrecognized layer type")
		}
	}
	return &network{newLayers}, nil
}

type network struct {
	layers []layers.Layer
}
