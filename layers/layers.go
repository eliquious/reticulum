package layers

import "github.com/eliquious/reticulum/volume"

// LayerType describes the network layer
type LayerType string

// LayerType enums
const (
	FullyConnected    LayerType = "fc"
	LocalResponseNorm LayerType = "lrn"
	Dropout           LayerType = "dropout"
	Input             LayerType = "input"
	SoftMax           LayerType = "softmax"
	Regression        LayerType = "regression"
	Conv              LayerType = "conv"
	Pool              LayerType = "pool"
	Relu              LayerType = "relu"
	Sigmoid           LayerType = "sigmoid"
	Tanh              LayerType = "tanh"
	Maxout            LayerType = "maxout"
	SVM               LayerType = "svm"
)

// Layer represents a layer in the neural network.
type Layer interface {
	Forward(vol *volume.Volume, training bool) *volume.Volume
	Backward()
	GetResponse() []LayerResponse
}

// LossLayer extends the Layer interface with the Loss function
type LossLayer interface {
	Layer
	Loss(index int)
}

// LayerResponse represents the layer parameters (weights) and gradients.
type LayerResponse struct {
	Weights    []float64
	Gradients  []float64
	L1DecayMul float64
	L2DecayMul float64
}
