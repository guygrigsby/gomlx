// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
)

// NodeTypeOptimizationBarrier is the NodeType for OptimizationBarrier operations.
// (1002 = While, 1003 = If; these manual NodeTypes live outside the generated range.)
const NodeTypeOptimizationBarrier NodeType = 1004

// nodeInputsOptimizationBarrier holds the inputs for an OptimizationBarrier op.
type nodeInputsOptimizationBarrier struct {
	operands []*Node
}

// Type implements NodeInputs.
func (ni *nodeInputsOptimizationBarrier) Type() NodeType { return NodeTypeOptimizationBarrier }

// String implements NodeInputs.
func (ni *nodeInputsOptimizationBarrier) String() string { return "OptimizationBarrier" }

// OptimizationBarrier returns its operands unchanged, but forbids the compiler from
// optimizing across the barrier. In particular it prevents common-subexpression
// elimination from merging the operand subgraphs with identical computations
// elsewhere in the graph.
//
// This is the building block for gradient checkpointing / rematerialization: a
// recomputed forward built from barriered inputs stays a distinct subgraph from the
// original forward, so the original's (large) intermediate activations are dead
// once its output is produced and the backend can free them — recomputing them
// only in the backward pass.
//
// Pass two or more operands together to force a real barrier: with a single operand
// some backends treat it as a no-op (a value already depends on itself). The
// gradient is the identity (each input gets its corresponding output's adjoint).
//
// Returns one node per operand, in order.
func OptimizationBarrier(operands ...*Node) []*Node {
	if len(operands) == 0 {
		exceptions.Panicf("OptimizationBarrier requires at least one operand")
	}
	g := operands[0].graph
	g.AssertBuilding()
	validateBuildingGraphFromInputs(operands...)

	inputValues := make([]compute.Value, len(operands))
	for i, o := range operands {
		inputValues[i] = o.outputOps[0]
	}
	results, err := g.currentFunc.backendFunc.OptimizationBarrier(inputValues...)
	if err != nil {
		panic(errors.WithMessage(err, "OptimizationBarrier operation failed"))
	}

	outputShapes := make([]shapes.Shape, len(results))
	outputOps := make([]compute.Value, len(results))
	for i, res := range results {
		outputShapes[i] = mustNoError(g.builder.OpShape(res))
		outputOps[i] = res
	}

	node := &Node{
		graph:        g,
		outputOps:    outputOps,
		outputShapes: outputShapes,
		inputs:       &nodeInputsOptimizationBarrier{operands: operands},
		inputNodes:   operands,
		scope:        g.currentFunc,
	}
	g.registerNode(node)
	return splitNode(node)
}

// optimizationBarrierVJP is the identity: the barrier passes values through
// unchanged, so each operand's gradient is its corresponding output's adjoint.
func optimizationBarrierVJP(node *Node, vjpForOutputs []*Node, _ shapes.Shape) []*Node {
	return vjpForOutputs
}

func init() {
	VJPRegistration[NodeTypeOptimizationBarrier] = optimizationBarrierVJP
}
