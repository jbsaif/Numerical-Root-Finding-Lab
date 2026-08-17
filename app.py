import io
import base64
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sympy as sp
from flask import Flask, render_template, request, send_file
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)

# Safe Math Evaluator Function
def parse_equation(eq_str):
    """
    Parses a math expression string safely and returns an executable Python function f(x).
    """
    x = sp.Symbol('x')
    clean_eq = eq_str.replace('^', '**')
    expr = sp.sympify(clean_eq)
    f = sp.lambdify(x, expr, modules=['numpy', 'math'])
    return f, expr

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/bisection', methods=['GET', 'POST'])
def bisection():
    result = None
    steps = []
    steps_json = "[]"
    plot_url = None
    error = None
    equation = ""
    a = ""
    b = ""
    tol = 0.0001
    max_iter = 20

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

            if not equation or a is None or b is None:
                error = "Please provide an equation and both lower (a) and upper (b) bounds."
            else:
                f, expr = parse_equation(equation)
                fa = float(f(a))
                fb = float(f(b))

                if fa * fb >= 0:
                    error = f"The function must have opposite signs at endpoints. f(a)={fa:.4f}, f(b)={fb:.4f}"
                else:
                    c = a
                    a_curr, b_curr = a, b
                    for i in range(1, max_iter + 1):
                        c = (a_curr + b_curr) / 2.0
                        fc = float(f(c))
                        fa_curr = float(f(a_curr))

                        steps.append({
                            'iteration': i,
                            'a': f"{a_curr:.6f}",
                            'b': f"{b_curr:.6f}",
                            'c': f"{c:.6f}",
                            'fc': f"{fc:.6f}"
                        })

                        if abs(fc) < 1e-15 or (b_curr - a_curr) / 2.0 < tol:
                            break

                        if fa_curr * fc < 0:
                            b_curr = c
                        else:
                            a_curr = c

                    result = f"{c:.6f}"
                    steps_json = json.dumps(steps)

                    # Plotting
                    plt.figure(figsize=(6, 4))
                    x_vals = np.linspace(a, b, 400)
                    try:
                        y_vals = f(x_vals)
                    except Exception:
                        y_vals = [float(f(v)) for v in x_vals]

                    plt.plot(x_vals, y_vals, label=f'f(x) = {equation}', color='#38bdf8')
                    plt.axhline(0, color='gray', linestyle='--')
                    plt.scatter([float(result)], [float(f(float(result)))], color='#f43f5e', zorder=5, label=f'Root: {result}')
                    plt.title('Bisection Method Graph', color='white')
                    plt.xlabel('x', color='white')
                    plt.ylabel('f(x)', color='white')
                    plt.gca().set_facecolor('#0f172a')
                    plt.gcf().patch.set_facecolor('#1e293b')
                    plt.tick_params(colors='white')
                    plt.legend()
                    plt.tight_layout()

                    img = io.BytesIO()
                    plt.savefig(img, format='png', facecolor=plt.gcf().get_facecolor(), edgecolor='none')
                    img.seek(0)
                    plot_url = base64.b64encode(img.getvalue()).decode()
                    plt.close()

        except Exception as e:
            error = f"Invalid input or equation syntax: {str(e)}"

    return render_template('bisection.html', result=result, steps=steps, steps_json=steps_json, 
                           plot_url=plot_url, error=error, equation=equation, a=a, b=b, tol=tol, max_iter=max_iter)

@app.route('/false-position', methods=['GET', 'POST'])
def false_position():
    result = None
    steps = []
    steps_json = "[]"
    plot_url = None
    error = None
    equation = ""
    a = ""
    b = ""
    tol = 0.0001
    max_iter = 20

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

            if not equation or a is None or b is None:
                error = "Please provide an equation and both lower (a) and upper (b) bounds."
            else:
                f, expr = parse_equation(equation)
                fa = float(f(a))
                fb = float(f(b))

                if fa * fb >= 0:
                    error = f"The function must have opposite signs at endpoints. f(a)={fa:.4f}, f(b)={fb:.4f}"
                else:
                    c = a
                    a_curr, b_curr = a, b
                    for i in range(1, max_iter + 1):
                        fa_curr = float(f(a_curr))
                        fb_curr = float(f(b_curr))
                        
                        if (fa_curr - fb_curr) == 0:
                            break
                            
                        c = b_curr - (fb_curr * (a_curr - b_curr)) / (fa_curr - fb_curr)
                        fc = float(f(c))

                        steps.append({
                            'iteration': i,
                            'a': f"{a_curr:.6f}",
                            'b': f"{b_curr:.6f}",
                            'c': f"{c:.6f}",
                            'fc': f"{fc:.6f}"
                        })

                        if abs(fc) < 1e-15 or abs(b_curr - a_curr) < tol:
                            break

                        if fa_curr * fc < 0:
                            b_curr = c
                        else:
                            a_curr = c

                    result = f"{c:.6f}"
                    steps_json = json.dumps(steps)

                    # Plotting
                    plt.figure(figsize=(6, 4))
                    x_vals = np.linspace(a, b, 400)
                    try:
                        y_vals = f(x_vals)
                    except Exception:
                        y_vals = [float(f(v)) for v in x_vals]

                    plt.plot(x_vals, y_vals, label=f'f(x) = {equation}', color='#38bdf8')
                    plt.axhline(0, color='gray', linestyle='--')
                    plt.scatter([float(result)], [float(f(float(result)))], color='#f43f5e', zorder=5, label=f'Root: {result}')
                    plt.title('False Position Method Graph', color='white')
                    plt.xlabel('x', color='white')
                    plt.ylabel('f(x)', color='white')
                    plt.gca().set_facecolor('#0f172a')
                    plt.gcf().patch.set_facecolor('#1e293b')
                    plt.tick_params(colors='white')
                    plt.legend()
                    plt.tight_layout()

                    img = io.BytesIO()
                    plt.savefig(img, format='png', facecolor=plt.gcf().get_facecolor(), edgecolor='none')
                    img.seek(0)
                    plot_url = base64.b64encode(img.getvalue()).decode()
                    plt.close()

        except Exception as e:
            error = f"Invalid input or equation syntax: {str(e)}"

    return render_template('false_position.html', result=result, steps=steps, steps_json=steps_json, 
                           plot_url=plot_url, error=error, equation=equation, a=a, b=b, tol=tol, max_iter=max_iter)

@app.route('/download-pdf', methods=['POST'])
def download_pdf():
    method_name = request.form.get('method_name', 'Numerical Method')
    equation = request.form.get('equation', '')
    a = request.form.get('a', '')
    b = request.form.get('b', '')
    tol = request.form.get('tol', '')
    max_iter = request.form.get('max_iter', '')

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=15,
        alignment=1
    )

    normal_style = styles['Normal']

    elements.append(Paragraph(f"{method_name} - Execution Report", title_style))
    elements.append(Spacer(1, 10))

    meta_text = f"<b>Equation:</b> f(x) = {equation} <br/><b>Interval:</b> [{a}, {b}] <br/><b>Tolerance:</b> {tol} <br/><b>Max Iterations:</b> {max_iter}"
    elements.append(Paragraph(meta_text, normal_style))
    elements.append(Spacer(1, 15))

    doc.build(elements)
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name=f"{method_name.lower().replace(' ', '_')}_report.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)