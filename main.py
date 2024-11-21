from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.optimize import linprog
from typing import List, Optional
import numpy as np

app = FastAPI(
    title="Otimização Linear - Método Simplex",
    description="API para resolver problemas de programação linear usando o método Simplex",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimplexInput(BaseModel):
    objective: List[float]
    lhs_ineq: List[List[float]]
    rhs_ineq: List[float]
    desired_variations: List[float]

    class Config:
        schema_extra = {
            "example": {
                "objective": [3, 2],
                "lhs_ineq": [
                    [2, 1],
                    [1, 3],
                ],
                "rhs_ineq": [100, 90],
                "desired_variations": [10, 5],
            }
        }


class SimplexResult(BaseModel):
    status: int
    message: str
    optimal_value: Optional[float] = None
    solution: Optional[List[float]] = None
    shadow_prices: Optional[List[float]] = None
    variation_viable: Optional[List[bool]] = None
    new_optimal_values: Optional[List[Optional[float]]] = None


def calculate_shadow_prices(objective, lhs_ineq, rhs_ineq, original_result):
    shadow_prices = []
    delta = 1e-6

    for i in range(len(rhs_ineq)):
        perturbed_rhs = rhs_ineq.copy()
        perturbed_rhs[i] += delta

        perturbed_result = linprog(
            c=[-x for x in objective],
            A_ub=lhs_ineq,
            b_ub=perturbed_rhs,
            method="simplex",
        )

        if perturbed_result.success:
            shadow_price = (-perturbed_result.fun - (-original_result.fun)) / delta
            shadow_prices.append(shadow_price)
        else:
            shadow_prices.append(0.0)

    return shadow_prices


@app.post("/solve", response_model=SimplexResult)
async def solve_simplex(input_data: SimplexInput):
    try:
        objective_coeffs = [-x for x in input_data.objective]

        result = linprog(
            c=objective_coeffs,
            A_ub=input_data.lhs_ineq,
            b_ub=input_data.rhs_ineq,
            method="simplex",
        )

        if not result.success:
            raise HTTPException(
                status_code=400, detail=f"Otimização falhou: {result.message}"
            )

        shadow_prices = calculate_shadow_prices(
            input_data.objective, input_data.lhs_ineq, input_data.rhs_ineq, result
        )

        adjusted_rhs = [
            r + v for r, v in zip(input_data.rhs_ineq, input_data.desired_variations)
        ]

        varied_result = linprog(
            c=objective_coeffs,
            A_ub=input_data.lhs_ineq,
            b_ub=adjusted_rhs,
            method="simplex",
        )

        variation_viable = varied_result.success
        new_optimal_value = -varied_result.fun if varied_result.success else None

        return SimplexResult(
            status=1,
            message="Otimização concluída com sucesso",
            optimal_value=-result.fun,
            solution=result.x.tolist(),
            shadow_prices=shadow_prices,
            variation_viable=[variation_viable] * len(input_data.rhs_ineq),
            new_optimal_values=(
                [new_optimal_value] if new_optimal_value is not None else None
            ),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
