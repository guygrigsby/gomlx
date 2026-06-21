// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
)

// OptimizationBarrier returns its operands unchanged (the barrier only constrains
// the compiler, not the values).
func TestOptimizationBarrier_Identity(t *testing.T) {
	graphtest.RunTestGraphFn(t, "OptimizationBarrier: identity values",
		func(g *Graph) (inputs, outputs []*Node) {
			x := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 3)))       // [1 2 3]
			y := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 2, 2)), 0.5) // [[.5 .5][.5 .5]]
			inputs = []*Node{x, y}
			b := OptimizationBarrier(x, y)
			outputs = []*Node{b[0], b[1]}
			return
		}, []any{
			[]float32{1, 2, 3},
			[][]float32{{0.5, 0.5}, {0.5, 0.5}},
		}, 1e-6)
}

// The gradient flows through OptimizationBarrier as the identity, so gradients of a
// loss are unchanged by inserting a barrier on the inputs.
func TestOptimizationBarrier_Gradient(t *testing.T) {
	graphtest.RunTestGraphFn(t, "OptimizationBarrier: identity gradient",
		func(g *Graph) (inputs, outputs []*Node) {
			x := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 4))) // [1 2 3 4]
			y := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 4))) // [1 2 3 4]
			b := OptimizationBarrier(x, y)
			// loss = sum(b0 * b1); d/dx = y, d/dy = x.
			loss := ReduceAllSum(Mul(b[0], b[1]))
			grads := Gradient(loss, x, y)
			inputs = []*Node{x, y}
			outputs = append([]*Node{loss}, grads...)
			return
		}, []any{
			float32(1 + 4 + 9 + 16), // sum(x*y) with x==y
			[]float32{1, 2, 3, 4},   // d/dx = y
			[]float32{1, 2, 3, 4},   // d/dy = x
		}, 1e-6)
}
