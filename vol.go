package reticulum

// VolumeOptions stores volume options
type VolumeOptions struct {
	Zero            bool
	HasInitialValue bool
	InitialValue    float64
}

// VolumeOptionFunc modifies the VolumeOptions when creating a new Volume.
type VolumeOptionFunc func(*VolumeOptions)

// WithInitialValue sets the initial values of the Volume.
func WithInitialValue(value float64) VolumeOptionFunc {
	return func(opts *VolumeOptions) {
		opts.HasInitialValue = true
		opts.InitialValue = value
	}
}

// WithZeroes sets the initial values of the Volume to zero.
func WithZeroes(value float64) VolumeOptionFunc {
	return func(opts *VolumeOptions) {
		opts.HasInitialValue = true
		opts.Zero = true
	}
}

// NewVolume creates a new Volume of the given size and options.
func NewVolume(sx, sy, depth int, optFuncs ...VolumeOptionFunc) *Volume {
	n := sx * sy * depth
	w := make([]float64, 0, n)
	dw := make([]float64, 0, n)

	// Update opts
	opts := &VolumeOptions{}
	for _, optFn := range optFuncs {
		optFn(opts)
	}

	// Initialize weights
	if opts.HasInitialValue {
		if !opts.Zero {
			for i := 0; i < n; i++ {
				w[i] = opts.InitialValue
			}
		} else {
			// Arrays already contain zeroes.
		}
	} else {

		// weight normalization is done to equalize the output
		// variance of every neuron, otherwise neurons with a lot
		// of incoming connections have outputs of larger variance
		scale := math.Sqrt(1.0 / float64(n))
		for i := 0; i < n; i++ {
			w[i] = rand.Float64() * scale
		}
	}

	return &Volume{
		sx, sy, depth, n, w, dw,
	}
}

// Volume is the basic building block of all the data in a network.
// It is essentially a 3D block of numbers with a width (sx), height (sy),
// and a depth (depth). It is used to hold data for all the filters, volumes,
// weights and gradients w.r.t. the data.
type Volume struct {
	sx    int
	sy    int
	depth int
	n     int
	w     []float64
	dw    []float64
}

// Get returns a weight for the given position in the Volume.
func (v *Volume) Get(x, y, d int) {
	return v.w[v.getIndex(x, y, d)]
}

// Set updates the position in the Volume.
func (v *Volume) Set(x, y, d int, val float64) {
	v.w[v.getIndex(x, y, d)] = val
}

// Add adds the given value to the weight for the given position.
func (v *Volume) Add(x, y, d int, val float64) {
	v.w[v.getIndex(x, y, d)] += val
}

// getIndex returns the array index for the given position.
func (v *Volume) getIndex(x, y, d int) int {
	return ((v.sx*y)+x)*v.depth + d
}

// GetGrad returns the gradient at the given position.
func (v *Volume) GetGrad(x, y, d int) {
	return v.dw[v.getIndex(x, y, d)]
}

// SetGrad updates the gradient at the given position.
func (v *Volume) SetGrad(x, y, d int, val float64) {
	v.dw[v.getIndex(x, y, d)] = val
}

// AddGrad adds the given value to the gradient for the given position.
func (v *Volume) AddGrad(x, y, d int, val float64) {
	v.dw[v.getIndex(x, y, d)] += val
}

// Clone creates a new Volume with cloned weights and zeroed gradients.
func (v *Volume) Clone() *Volume {
	vol := NewVolume(v.sx, v.sy, v.depth, WithZeros(0.0))
	copy(vol.w, v.w)
	return vol
}

// CloneAndZero creates a Volume of the same size but with zero weights and gradients.
func (v *Volume) CloneAndZero() *Volume {
	return NewVolume(v.sx, v.sy, v.depth, WithZeros(0.0))
}

// AddFrom adds the weights from another Volume.
func (v *Volume) AddFrom(vol *Volume) {
	for i := 0; i < v.n; i++ {
		v.w[i] += vol.w[i]
	}
}

// AddFromScaled adds the weights from another Volume and scaled with the given value.
func (v *Volume) AddFromScaled(vol *Volume, scale float64) {
	for i := 0; i < v.n; i++ {
		v.w[i] += vol.w[i] * scale
	}
}

// SetConst sets the weights to the given value.
func (v *Volume) SetConst(val float64) {
	for i := 0; i < n; i++ {
		v.w[i] = val
	}
}
