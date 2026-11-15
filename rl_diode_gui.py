"""Streamlit calculator for a half-wave rectifier with an RL load and opposing DC source.

Assumptions:
- Ideal diode (zero forward voltage drop).
- Series R-L-DC circuit, with Vdc opposing the AC source.
- Sinusoidal AC source: v_ac(t) = sqrt(2) * Vac_rms * sin(2*pi*f*t).
- Periodic steady-state operation.

-- by Bahaa Alsaeed
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.integrate import quad
from scipy.optimize import brentq


@dataclass(frozen=True)
class CircuitResults:
    vm: float
    omega: float
    period: float
    xl: float
    z: float
    phi: float
    im: float
    tau: float
    alpha: float
    beta: float
    t_alpha: float
    t_beta: float
    cycle_count: int
    pattern_period: float
    k_constant: float
    i_avg: float
    i_rms: float
    p_resistor: float
    p_dc: float
    p_ac: float
    power_factor: float


def calculate_circuit(
    resistance: float,
    inductance_h: float,
    dc_voltage: float,
    ac_rms_voltage: float,
    frequency_hz: float,
) -> CircuitResults | None:
    """Calculate conduction interval, current metrics, powers, and true PF.

    Returns None when the AC peak voltage cannot overcome the opposing DC source.
    Raises ValueError if a numerical turn-off point cannot be found.
    """
    vm = math.sqrt(2.0) * ac_rms_voltage
    omega = 2.0 * math.pi * frequency_hz
    period = 1.0 / frequency_hz

    # With an ideal diode, conduction can start only if the AC peak exceeds Vdc.
    if vm <= dc_voltage:
        return None

    xl = omega * inductance_h
    z = math.hypot(resistance, xl)
    phi = math.atan2(xl, resistance)
    im = vm / z
    tau = inductance_h / resistance

    # First rising crossing of v_ac = Vdc.
    alpha = math.asin(dc_voltage / vm)
    t_alpha = alpha / omega

    # Exact solution of L di/dt + R i = Vm*sin(omega*t) - Vdc,
    # with i(t_alpha) = 0.
    k_constant = dc_voltage / resistance - im * math.sin(alpha - phi)

    def conducting_current(theta: float) -> float:
        time_s = theta / omega
        return (
            im * math.sin(theta - phi)
            - dc_voltage / resistance
            + k_constant * math.exp(-resistance / inductance_h * (time_s - t_alpha))
        )

    # Search for the first positive-to-zero crossing after turn-on.
    # The horizon scales with the electrical period and the RL time constant.
    max_angle = alpha + max(20.0 * 2.0 * math.pi, 20.0 * omega * tau + 4.0 * math.pi)
    step = 0.02
    left = alpha + 1e-5
    f_left = conducting_current(left)
    beta = None

    theta = left + step
    while theta <= max_angle:
        f_right = conducting_current(theta)
        if f_left > 0.0 and f_right <= 0.0:
            beta = brentq(conducting_current, left, theta, xtol=1e-13, rtol=1e-13)
            break
        left, f_left = theta, f_right
        theta += step

    if beta is None:
        raise ValueError(
            "Could not find the end of the conduction interval within the search range."
            " Try different practical component values."
        )

    t_beta = beta / omega

    # A pulse may, for unusual component values, extend beyond one AC cycle.
    # The next pulse begins at the first rising threshold crossing after turn-off.
    cycle_count = max(1, math.ceil((beta - alpha) / (2.0 * math.pi)))
    pattern_period = cycle_count * period
    full_pattern_angle = 2.0 * math.pi * cycle_count

    # Integrate only over the conducting interval; current is zero elsewhere.
    charge_area = quad(conducting_current, alpha, beta, epsabs=1e-10, epsrel=1e-10)[0]
    square_area = quad(
        lambda x: conducting_current(x) ** 2,
        alpha,
        beta,
        epsabs=1e-10,
        epsrel=1e-10,
    )[0]
    ac_power_area = quad(
        lambda x: vm * math.sin(x) * conducting_current(x),
        alpha,
        beta,
        epsabs=1e-10,
        epsrel=1e-10,
    )[0]

    i_avg = charge_area / full_pattern_angle
    i_rms = math.sqrt(square_area / full_pattern_angle)
    p_resistor = resistance * i_rms**2
    p_dc = dc_voltage * i_avg
    p_ac = ac_power_area / full_pattern_angle
    power_factor = p_ac / (ac_rms_voltage * i_rms) if i_rms > 0 else float("nan")

    return CircuitResults(
        vm=vm,
        omega=omega,
        period=period,
        xl=xl,
        z=z,
        phi=phi,
        im=im,
        tau=tau,
        alpha=alpha,
        beta=beta,
        t_alpha=t_alpha,
        t_beta=t_beta,
        cycle_count=cycle_count,
        pattern_period=pattern_period,
        k_constant=k_constant,
        i_avg=i_avg,
        i_rms=i_rms,
        p_resistor=p_resistor,
        p_dc=p_dc,
        p_ac=p_ac,
        power_factor=power_factor,
    )


def make_plot(
    result: CircuitResults,
    resistance: float,
    inductance_h: float,
    dc_voltage: float,
    ac_rms_voltage: float,
    frequency_hz: float,
):
    """Plot AC voltage, opposing DC voltage, and circuit current over one pattern period."""
    samples = max(2500, 1800 * result.cycle_count)
    time = np.linspace(0.0, result.pattern_period, samples)
    theta = result.omega * time
    voltage_ac = result.vm * np.sin(theta)

    current_values = np.zeros_like(theta)
    conducting = (theta >= result.alpha) & (theta <= result.beta)
    t_alpha = result.t_alpha
    current_values[conducting] = (
        result.im * np.sin(theta[conducting] - result.phi)
        - dc_voltage / resistance
        + result.k_constant
        * np.exp(-resistance / inductance_h * (time[conducting] - t_alpha))
    )

    fig, voltage_axis = plt.subplots(figsize=(10, 4.8))
    current_axis = voltage_axis.twinx()

    voltage_axis.plot(time * 1000.0, voltage_ac, label="AC voltage", linewidth=1.7)
    voltage_axis.axhline(dc_voltage, linestyle="--", linewidth=1.3, label="DC voltage magnitude", color='green')
    current_axis.plot(time * 1000.0, current_values, label="Current", linewidth=2.0, color='red')

    voltage_axis.set_xlabel("Time (ms)")
    voltage_axis.set_ylabel("Voltage (V)")
    current_axis.set_ylabel("Current (A)")
    voltage_axis.set_title("Voltage and Current over the Repetition Period")
    voltage_axis.grid(True, alpha=0.3)

    lines_1, labels_1 = voltage_axis.get_legend_handles_labels()
    lines_2, labels_2 = current_axis.get_legend_handles_labels()
    voltage_axis.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    fig.tight_layout()
    return fig


# ------------------------------------
# Streamlit page
# ------------------------------------
st.set_page_config(page_title="RL Diode Circuit Calculator", page_icon="⚡", layout="wide")
st.title("RL Diode Circuit Calculator")
st.caption(
    "Current, power,"
    " and power factor analysis of a half-wave rectifier with an RL load and an opposing DC source.")

with st.sidebar:
    st.header("Circuit Parameters")
    with st.form("circuit_inputs"):
        resistance = st.number_input(
            "Resistance R (Ω)", min_value=0.001, value=2.0, step=0.1, format="%.4f")
        inductance_mh = st.number_input(
            "Inductance L (mH)", min_value=0.001, value=20.0, step=1.0, format="%.4f")
        dc_voltage = st.number_input(
            "DC Source Voltage Vdc (V)", min_value=0.0, value=100.0, step=5.0, format="%.4f")
        ac_rms_voltage = st.number_input(
            "AC Source Voltage (RMS) (V)", min_value=0.001, value=120.0, step=5.0, format="%.4f")
        frequency_hz = st.number_input(
            "Frequency f (Hz)", min_value=0.001, value=60.0, step=1.0, format="%.4f")
        submitted = st.form_submit_button("Calculate", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Assumptions**")
    st.markdown(
        "- Ideal diode.\n"
        "- DC source opposes the AC source.\n"
        "- Periodic steady-state operation."
    )

if "last_submitted" not in st.session_state:
    st.session_state.last_submitted = True
if submitted:
    st.session_state.last_submitted = True

if st.session_state.last_submitted:
    inductance_h = inductance_mh / 1000.0

    try:
        result = calculate_circuit(
            resistance=resistance,
            inductance_h=inductance_h,
            dc_voltage=dc_voltage,
            ac_rms_voltage=ac_rms_voltage,
            frequency_hz=frequency_hz,
        )

        if result is None:
            st.warning(
                "The diode does not conduct with these values because"
                " the AC peak voltage does not exceed the opposing DC source voltage. "
                f"قمة AC = {math.sqrt(2) * ac_rms_voltage:.4f} V، بينما Vdc = {dc_voltage:.4f} V."
            )
        else:
            st.subheader("Calculated Results")

            # (a) Current expression
            with st.container(border=True):
                st.markdown("### (a) Current Expression i(t)")
                st.latex(
                    r"i(t)=I_m\sin(\omega t-\phi)-\frac{V_{dc}}{R}"
                    r"+K e^{-\frac{R}{L}(t-t_\alpha)}"
                )
                st.latex(
                    rf"i(t)= {result.im:.5f}\sin({result.omega:.5f}t-"
                    rf"{result.phi:.5f})-{dc_voltage / resistance:.5f}"
                    rf"+{result.k_constant:.5f}e^{{-{resistance /
                                                     inductance_h:.5f}(t-{result.t_alpha:.7f})}}"
                )
                st.markdown(
                    "This expression is valid only while the diode is conducting."
                    " The current is zero outside the conduction interval."
                )
                st.latex(
                    rf"t_\alpha={result.t_alpha * 1000:.5f}\;ms"
                    rf",\qquad t_\beta={result.t_beta * 1000:.5f}\;ms"
                )
                st.latex(
                    rf"\alpha={math.degrees(result.alpha):.4f}^\circ"
                    rf",\qquad \beta={math.degrees(result.beta):.4f}^\circ"
                )
                st.latex(
                    rf"t_\alpha+kT_p\leq t\leq t_\beta+kT_p"
                    rf",\quad T_p={result.pattern_period * 1000:.5f}\;ms"
                )
                st.caption(
                    f"The current pulse repeats every {result.cycle_count} AC cycle(s); k is an integer."
                )

            # (b) Resistor power
            with st.container(border=True):
                st.markdown("### (b) Resistor Power P_R")
                st.latex(r"P_R=R I_{rms}^{2}")
                st.metric("Resistor Power", f"{result.p_resistor:.4f} W")

            # (c) DC source power
            with st.container(border=True):
                st.markdown("### (c) DC Source Absorbed Power")
                st.latex(r"P_{dc}=V_{dc} I_{avg}")
                st.metric("DC Source Absorbed Power", f"{result.p_dc:.4f} W")
                st.caption("The positive sign indicates that the DC source absorbs power"
                           " under the assumed polarity.")

            # (d) AC power and power factor
            with st.container(border=True):
                st.markdown("### (d) AC Source Power and Power Factor")
                st.latex(r"P_{ac}=\frac{1}{T_p}\int_0^{T_p}v_{ac}(t)i(t)\,dt")
                col_a, col_b = st.columns(2)
                col_a.metric("AC Source Supplied Power", f"{result.p_ac:.4f} W")
                col_b.metric("True Power Factor (PF)", f"{result.power_factor:.5f}")
                st.latex(r"PF=\frac{P_{ac}}{V_{ac,rms}I_{rms}}")

            st.subheader("Additional Values")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Average Current", f"{result.i_avg:.5f} A")
            m2.metric("RMS Current", f"{result.i_rms:.5f} A")
            m3.metric("Inductive Reactance", f"{result.xl:.5f} Ω")
            m4.metric("Impedance Phase Angle φ", f"{math.degrees(result.phi):.3f}°")

            st.subheader("Waveforms")
            figure = make_plot(
                result,
                resistance,
                inductance_h,
                dc_voltage,
                ac_rms_voltage,
                frequency_hz,
            )
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)

            st.subheader("Power Balance Verification")
            energy_error = result.p_ac - (result.p_resistor + result.p_dc)
            st.write(
                f"P_ac = {result.p_ac:.6f} W, while P_R + P_dc = "
                f"{result.p_resistor + result.p_dc:.6f} W."
            )
            st.success(f"Power balance error = {energy_error:.3e} W")

    except (ValueError, OverflowError) as error:
        st.error(str(error))
else:
    st.info("Enter the circuit parameters in the sidebar, then click Calculate.")

st.divider()
st.caption("This model assumes an ideal diode and does not include diode forward voltage,"
           " source resistance, or inductor losses.")
