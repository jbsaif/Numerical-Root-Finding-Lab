from flask import Flask, render_template, request, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import json
import sympy as sp
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)

def parse_equation(eq_str):
    x = sp.Symbol('x')
    eq_str = eq_str.replace('^', '**')
    expr = sp.sympify(eq_str)
    f = sp.lambdify(x, expr, 'numpy')
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
        equation = request.form.get('equation')
        try:
            a = float(request.form.get('a'))
            b = float(request.form.get('b'))
            tol = float(request.form.get('tol'))
            max_iter = int(request.form.get('max_iter'))

            f, expr = parse_equation(equation)

            if f(a) * f(b) >= 0:
                error = "The function must have opposite signs at endpoints 'a' and 'b'."
            else:
                c = a
                for i in range(1, max_iter + 1):
                    c = (a + b) / 2.0
                    fc = f(c)
                    fa = f(a)
                    
                    steps.append({
                        'iteration': i,
                        'a': f"{a:.6f}",
                        'b': f"{b:.6f}",
                        'c': f"{c:.6f}",
                        'fc': f"{fc:.6f}"
                    })

                    if abs(fc) < 1e-15 or (b - a) / 2.0 < tol:
                        break

                    if f(a) * fc < 0:
                        b = c
                    else:
                        a = c

                result = f"{c:.6f}"
                steps_json = json.dumps(steps)

                plt.figure(figsize=(6, 4))
                x_vals = np.linspace(float(request.form.get('a')), float(request.form.get('b')), 400)
                y_vals = f(x_vals)
                plt.plot(x_vals, y_vals, label=f'f(x) = {equation}', color='#38bdf8')
                plt.axhline(0, color='gray', linestyle='--')
                plt.scatter([float(result)], [f(float(result))], color='#f43f5e', zorder=5, label=f'Root: {result}')
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
        equation = request.form.get('equation')
        try:
            a = float(request.form.get('a'))
            b = float(request.form.get('b'))
            tol = float(request.form.get('tol'))
            max_iter = int(request.form.get('max_iter'))

            f, expr = parse_equation(equation)

            if f(a) * f(b) >= 0:
                error = "The function must have opposite signs at endpoints 'a' and 'b'."
            else:
                c = a
                for i in range(1, max_iter + 1):
                    fa = f(a)
                    fb = f(b)
                    c = b - (fb * (a - b)) / (fa - fb)
                    fc = f(c)
                    
                    steps.append({
                        'iteration': i,
                        'a': f"{a:.6f}",
                        'b': f"{b:.6f}",
                        'c': f"{c:.6f}",
                        'fc': f"{fc:.6f}"
                    })

                    if abs(fc) < 1e-15 or abs(b - a) < tol:
                        break

                    if fa * fc < 0:
                        b = c
                    else:
                        a = c

                result = f"{c:.6f}"
                steps_json = json.dumps(steps)

                plt.figure(figsize=(6, 4))
                x_vals = np.linspace(float(request.form.get('a')), float(request.form.get('b')), 400)
                y_vals = f(x_vals)
                plt.plot(x_vals, y_vals, label=f'f(x) = {equation}', color='#38bdf8')
                plt.axhline(0, color='gray', linestyle='--')
                plt.scatter([float(result)], [f(float(result))], color='#f43f5e', zorder=5, label=f'Root: {result}')
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
    equation = request.form.get('equation')
    a = request.form.get('a')
    b = request.form.get('b')
    tol = request.form.get('tol')
    max_iter = request.form.get('max_iter')

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