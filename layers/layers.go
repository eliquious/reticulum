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
	ReLU              LayerType = "relu"
	Sigmoid           LayerType = "sigmoid"
	Tanh              LayerType = "tanh"
	Maxout            LayerType = "maxout"
	SVM               LayerType = "svm"
)

// LayerDef outlines the layer type, size and config.
type LayerDef struct {
	Type LayerType

	// Input dimensions
	Input volume.Dimensions

	// Output dim
	Output volume.Dimensions

	// LayerConfig contains layer specific requirements
	LayerConfig interface{}
}

// Layer represents a layer in the neural network.
type Layer interface {
	Forward(vol *volume.Volume, training bool) *volume.Volume
	Backward()
	GetResponse() []LayerResponse
}

// LossLayer extends the Layer interface with the Loss function
type LossLayer interface {
	Layer
	Loss(index int) float64
}

// RegressionLossLayer extends the Layer interface with the Loss function
type RegressionLossLayer interface {
	Layer
	MultiDimensionalLoss(losses []float64) float64
	DimensionalLoss(index int, value float64) float64
}

// LayerResponse represents the layer parameters (weights) and gradients.
type LayerResponse struct {
	Weights    []float64
	Gradients  []float64
	L1DecayMul float64
	L2DecayMul float64
}
