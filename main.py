from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.optimize import linprog
from typing import List, Optional

app = FastAPI()

class SimplexInput(BaseModel):
    objective: List[float]
    lhs_ineq: List[List[float]]
    rhs_ineq: List[float]
    desired_variations: List[float]

class SimplexResult(BaseModel):
    status: int
    message: str
    optimal_value: Optional[float] = None
    solution: Optional[List[float]] = None
    shadow_prices: Optional[List[float]] = None
    variation_viable: Optional[List[bool]] = None
    new_optimal_values: Optional[List[Optional[float]]] = None

@app.post("/solve", response_model=SimplexResult)
async def solve_simplex(input_data: SimplexInput):
    result = linprog(
        c=input_data.objective,
        A_ub=input_data.lhs_ineq,
        b_ub=input_data.rhs_ineq,
        method='simplex'
    )

    if not result.success:
        raise HTTPException(status_code=400, detail="Optimization failed: " + result.message)

    shadow_prices = []
    for i, constraint_rhs in enumerate(input_data.rhs_ineq):
        perturbed_rhs = input_data.rhs_ineq[:]
        perturbed_rhs[i] += 1e-5
        perturbed_result = linprog(
            c=input_data.objective,
            A_ub=input_data.lhs_ineq,
            b_ub=perturbed_rhs,
            method='simplex'
        )
        if perturbed_result.success:
            shadow_price = (perturbed_result.fun - result.fun) / 1e-5
            shadow_prices.append(shadow_price)
        else:
            shadow_prices.append(0.0)

    variation_viable = []
    adjusted_rhs = [rhs + variation for rhs, variation in zip(input_data.rhs_ineq, input_data.desired_variations)]
    print('Adjusted RHS with combined variations:', adjusted_rhs)

    adjusted_result = linprog(
        c=input_data.objective,
        A_ub=input_data.lhs_ineq,
        b_ub=adjusted_rhs,
        method='simplex'
    )

    if adjusted_result.success:
        variation_viable = [True] * len(input_data.desired_variations)
        new_optimal_value = adjusted_result.fun
    else:
        variation_viable = [False] * len(input_data.desired_variations)
        new_optimal_value = None


    combined_optimal_value = None
    if all(variation_viable):
        combined_rhs = [
            rhs + variation for rhs, variation in zip(input_data.rhs_ineq, input_data.desired_variations)
        ]
        print('combined_rhs:', combined_rhs)
        combined_result = linprog(
            c=input_data.objective,
            A_ub=input_data.lhs_ineq,
            b_ub=combined_rhs,
            method='simplex'
        )
        if combined_result.success:
            combined_optimal_value = combined_result.fun

    new_optimal_values = [combined_optimal_value] if combined_optimal_value is not None else []

    return SimplexResult(
        status=1,
        message="Optimization successful",
        optimal_value=result.fun,
        solution=result.x.tolist(),
        shadow_prices=shadow_prices,
        variation_viable=variation_viable,
        new_optimal_values=new_optimal_values
    )
