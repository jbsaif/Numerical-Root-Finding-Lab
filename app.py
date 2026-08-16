import io
import base64
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, make_pdf if False else None
from sympy import sympify, symbols, lambdify, diff

app = Flask(__name__)

# Safe Math Evaluator Function
def safe_eval_function(eq_str):
    """
    Parses a math expression string and returns a executable Python function f(x).
    """
    x = symbols('x')
    clean_eq = eq_str.replace('^', '**')
    expr = sympify(clean_eq)
    # Use numpy and math modules for numerical evaluation
    f = lambdify(x, expr, modules=['numpy', 'math'])
    return f, expr

# --- BISECTION METHOD ---
def solve_bisection(eq_str, a_val, b_val, tol_val, max_iter_val):
    try:
        f, expr = safe_eval_function(eq_str)
        a_val, b_val = float(a_val), float(b_val)
        tol_val = float(tol_val)
        max_iter_val = int(max_iter_val)

        fa = float(f(a_val))
        fb = float(f(b_val))

        if fa * fb > 0:
            return None, None, None, f"f(a)={fa:.4f} and f(b)={fb:.4f} must have opposite signs."

        steps = []
        c = a_val

        for i in range(1, max_iter_val + 1):
            c = (a_val + b_val) / 2.0
            fc = float(f(c))

            steps.append({
                'iteration': i,
                'a': round(a_val, 6),
                'b': round(b_val, 6),
                'c': round(c, 6),
                'fc': round(fc, 6)
            })

            if abs(fc) < tol_val or (b_val - a_val) / 2.0 < tol_val:
                break

            if float(f(a_val)) * fc < 0:
                b_val = c
            else:
                a_val = c

        # Plot Generation
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6, 4))
        x_min = min(float(a_val), float(b_val)) - 1.0
        x_max = max(float(a_val), float(b_val)) + 1.0
        x_vals = np.linspace(x_min, x_max, 400)
        
        try:
            y_vals = f(x_vals)
        except Exception:
            y_vals = [float(f(v)) for v in x_vals]

        ax.plot(x_vals, y_vals, label=f'f(x) = {eq_str}', color='#38bdf8', linewidth=2)
        ax.axhline(0, color='#64748b', linestyle='--', alpha=0.7)
        ax.axvline(c, color='#f43f5e', linestyle=':', label=f'Root ≈ {round(c, 4)}')
        ax.scatter([c], [f(c)], color='#f43f5e', s=50, zorder=5)

        ax.legend(facecolor='#1e293b', edgecolor='#334155')
        ax.grid(True, color='#334155', alpha=0.3)

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

        return round(c, 6), steps, plot_url, None

    except Exception as e:
        return None, None, None, f"Invalid input or equation syntax: {str(e)}"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/bisection', methods=['GET', 'POST'])
def bisection():
    if request.method == 'POST':
        equation = request.form.get('equation', '').strip()
        a_raw = request.form.get('a', '').strip()
        b_raw = request.form.get('b', '').strip()
        tol_raw = request.form.get('tol', '0.0001').strip()
        max_iter_raw = request.form.get('max_iter', '20').strip()

        try:
            a = float(a_raw) if a_raw != '' else None
            b = float(b_raw) if b_raw != '' else None
            tol = float(tol_raw) if tol_raw != '' else 0.0001
            max_iter = int(max_iter_raw) if max_iter_raw != '' else 20
        except ValueError:
            return render_template(
                'bisection.html',
                error="Please enter valid numerical values for bounds, tolerance, and max iterations.",
                equation=equation, a=None, b=None, tol=0.0001, max_iter=20,
                result=None, steps=[], steps_json="[]", plot_url=None
            )

        if not equation or a is None or b is None:
            return render_template(
                'bisection.html',
                error="Please provide an equation and both upper/lower bounds.",
                equation=equation, a=a, b=b, tol=tol, max_iter=max_iter,
                result=None, steps=[], steps_json="[]", plot_url=None
            )

        result, steps, plot_url, error = solve_bisection(equation, a, b, tol, max_iter)

        return render_template(
            'bisection.html',
            equation=equation,
            a=a,
            b=b,
            tol=tol,
            max_iter=max_iter,
            result=result,
            steps=steps,
            steps_json=json.dumps(steps) if steps else "[]",
            plot_url=plot_url,
            error=error
        )

    return render_template(
        'bisection.html',
        equation="",
        a=None,
        b=None,
        tol=0.0001,
        max_iter=20,
        result=None,
        steps=[],
        steps_json="[]",
        plot_url=None,
        error=None
    )


if __name__ == '__main__':
    app.run(debug=True)