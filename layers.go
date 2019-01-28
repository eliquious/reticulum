package reticulum

// Layer represents a layer in the neural network.
type Layer interface {
	Forward(vol *Volume, training bool) *Volume
	Backward()
	GetResponse() []LayerResponse
}

// LayerResponse represents the layer parameters (weights) and gradients.
type LayerResponse struct {
	Weights    []float64
	Gradients  []float64
	L1DecayMul float64
	L2DecayMul float64
}
