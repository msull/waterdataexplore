from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import curve_fit

# Set the page configuration
st.set_page_config(
    "Water Data Exploration", layout="wide", initial_sidebar_state="collapsed"
)


def main():
    """Main function to run the Streamlit app."""

    model_fit: Optional[FitResult] = None

    st.title("Water Data Exploration")
    raw_water_data = load_water_data()
    options = [
        raw_water_data.iloc[0]["date"].to_pydatetime(),
        raw_water_data.iloc[-1]["date"].to_pydatetime(),
    ]

    with st.expander("Data"):
        st.write(
            "Data for USGS 12080010 DESCHUTES RIVER AT E ST BRIDGE AT TUMWATER, WA"
        )
        st.write(
            "Sourced from https://waterdata.usgs.gov/nwis/dv?referred_module=sw&cb_00060=on&cb_00065=on&site_no=12080010"
        )

        st.dataframe(raw_water_data[["date", "stage_val", "discharge_rate"]])

    data_col, fit_col = st.columns((3, 1))

    with fit_col:
        with st.expander("Fit model to data", expanded=True):
            train_from = pd.to_datetime(
                st.date_input(
                    "Training Start Date",
                    options[0],
                    min_value=options[0],
                    max_value=options[1],
                )
            )
            train_to = pd.to_datetime(
                st.date_input(
                    "Training End Date",
                    options[1],
                    min_value=options[0],
                    max_value=options[1],
                )
            )
            if not train_to > train_from:
                st.error("Must select an end date after the start date to fit a mode")
            # slider_from, slider_to = st.slider("Training Dates", options[0], options[1], value=options)
            # train_from = pd.to_datetime(slider_from)
            # train_to = pd.to_datetime(slider_to)
            # del slider_from
            # del slider_to
            fit_type = st.selectbox(
                "Model Type", ("None", "Power Law", "Polynomial Model")
            )

            df = raw_water_data
            discharge_by_stage_water_data = df[
                (df["date"] >= train_from) & (df["date"] <= train_to)
            ]
            train_stage = discharge_by_stage_water_data["stage_val"].values
            train_discharge = discharge_by_stage_water_data["discharge_rate"].values

            if fit_type == "Power Law":
                model_fit = fit_power_law(
                    train_stage, train_discharge, train_from, train_to
                )
            elif fit_type == "Polynomial Model":
                degree = st.number_input(
                    "fit-degree", min_value=2, max_value=5, value=2
                )
                model_fit = fit_polynomial(
                    train_stage, train_discharge, degree, train_from, train_to
                )

            if model_fit:
                metrics = [
                    ("RMSE on training data", round(model_fit.rmse, 4)),
                ]
                for metric in metrics:
                    st.metric(*metric)
        if st.checkbox("View rating table for fit model"):
            with st.expander("Rating Table", expanded=True):
                if not model_fit:
                    st.info("Fit a model first")
                else:
                    min_stage = float(
                        raw_water_data["stage_val"].min() // 5 * 5
                    )  # nearest lower 5
                    min_val = round(raw_water_data["stage_val"].min(), 1)
                    max_stage = float(
                        raw_water_data["stage_val"].max() // 5 * 5 + 5
                    )  # nearest upper 5
                    columns = iter(st.columns(2))
                    with next(columns):
                        start = st.number_input(
                            "Start Stage Level",
                            min_value=min_stage - 25,
                            max_value=max_stage + 25,
                            value=min_val,
                            step=0.1,
                        )
                    with next(columns):
                        stop = st.number_input(
                            "Stop Stage Level",
                            min_value=min_stage - 25,
                            max_value=max_stage + 25,
                            value=max_stage,
                            step=0.1,
                        )

                    stages = np.arange(start, stop + 0.1, 0.1)
                    discharges = model_fit.predict(stages)
                    rating_table = pd.DataFrame(
                        {"Stage Level": stages, "Predicted Discharge": discharges}
                    )
                    st.dataframe(
                        rating_table.style.format(formatter="{:.2f}"), hide_index=True
                    )

    with data_col:
        with st.expander("Daily Stage and Discharge", expanded=True):
            # draw date select box
            slider_from, slider_to = st.slider(
                "Dates", options[0], options[1], value=options
            )
            filter_from = pd.to_datetime(slider_from)
            filter_to = pd.to_datetime(slider_to)
            del slider_from
            del slider_to

            df = raw_water_data
            daily_water_data = df[
                (df["date"] >= filter_from) & (df["date"] <= filter_to)
            ]

            view_type = st.selectbox(
                "View Option",
                ("Adjusted", "Raw", "Normalized"),
                label_visibility="collapsed",
            )
            normalize_data = view_type == "Normalized"

            daily_fig = go.Figure()
            stage_key = "stage_val"
            discharge_key = "discharge_rate"
            stage_range = None
            if view_type == "Adjusted":
                stage_range = [
                    int(df["stage_val"].min()) - 3,
                    int(df["stage_val"].max()) + 3,
                ]
            elif view_type == "Raw":
                pass
            elif view_type == "Normalized":
                stage_key += "_norm"
                discharge_key += "_norm"
                stage_range = None
            else:
                raise ValueError("Invalid view type")

            daily_fig.add_trace(
                go.Bar(
                    x=daily_water_data["date"],
                    y=daily_water_data[stage_key],
                    name="Stage",
                    marker_color="blue",
                )
            )

            daily_fig.add_trace(
                go.Scatter(
                    x=daily_water_data["date"],
                    y=daily_water_data[discharge_key],
                    name="Discharge",
                    line=dict(color="red"),
                    yaxis="y2",
                )
            )

            daily_fig.update_layout(
                # xaxis=dict(domain=[0.3, 0.7]),
                yaxis=dict(
                    title="Stage (f)",
                    titlefont=dict(color="blue"),
                    tickfont=dict(color="blue"),
                    range=stage_range,
                ),
                yaxis2=dict(
                    title="Discharge (f^3/s)",
                    titlefont=dict(color="red"),
                    tickfont=dict(color="red"),
                    overlaying="y",
                    side="right",
                ),
                legend=dict(yanchor="top", y=-0.3, xanchor="left", x=0.25),
            )
            # Predict discharge using the fitted model
            if model_fit:
                predicted_discharge = model_fit.predict(
                    daily_water_data["stage_val"], normalized=False
                )
                daily_display_rmse = round(
                    calculate_rmse(
                        daily_water_data["discharge_rate"], predicted_discharge
                    ),
                    4,
                )

                if normalize_data:
                    display_discharge = model_fit.predict(
                        daily_water_data["stage_val"], normalized=True
                    )
                else:
                    display_discharge = predicted_discharge

                # Add predicted discharge to the daily discharge plot

                daily_fig.add_trace(
                    go.Scatter(
                        x=daily_water_data["date"],
                        y=display_discharge,
                        name="Predicted Discharge",
                        line=dict(color="cyan"),
                        yaxis="y2",
                    )
                )
                st.metric("Model RMSE on displayed data", daily_display_rmse)

            st.plotly_chart(daily_fig, use_container_width=True)

        with st.expander("Discharge by Stage Level", expanded=True):
            discharge_fig = go.Figure()
            slider_from, slider_to = st.slider(
                "Dates", options[0], options[1], value=options, key="dates-2"
            )
            filter_from = pd.to_datetime(slider_from)
            filter_to = pd.to_datetime(slider_to)
            del slider_from
            del slider_to

            df = raw_water_data
            discharge_by_stage_water_data = df[
                (df["date"] >= filter_from) & (df["date"] <= filter_to)
            ]
            discharge_by_stage_stage = discharge_by_stage_water_data["stage_val"].values
            discharge_by_stage_discharge = discharge_by_stage_water_data[
                "discharge_rate"
            ].values
            discharge_fig.add_trace(
                go.Scatter(
                    x=discharge_by_stage_stage,
                    y=discharge_by_stage_discharge,
                    mode="markers",
                    name="Data",
                )
            )

            if model_fit is not None:
                discharge_fig.add_trace(
                    go.Scatter(
                        x=model_fit.stage_fit,
                        y=model_fit.discharge_fit,
                        mode="lines",
                        name=model_fit.label,
                    )
                )
                predicted_discharge = model_fit.predict(
                    discharge_by_stage_water_data["stage_val"], normalized=False
                )
                by_stage_rmse = round(
                    calculate_rmse(
                        discharge_by_stage_water_data["discharge_rate"],
                        predicted_discharge,
                    ),
                    4,
                )
                st.metric("Model RMSE on displayed data", by_stage_rmse)
                # fig.update_layout(xaxis_title="Stage (f)", yaxis_title="Discharge (f^3/s)", title="Stage Level and Discharge")

            discharge_fig.update_layout(
                xaxis_title="Stage (f)",
                yaxis_title="Discharge (f^3/s)",
                title="Discharge by stage level",
                legend=dict(
                    yanchor="top",
                    y=-0.3,
                    xanchor="left",
                    x=-0.1,  # adjust these as needed  # adjust these as needed
                ),
            )

            st.plotly_chart(discharge_fig, use_container_width=True)


def calculate_rmse(y_actual, y_pred):
    """Calculate Root Mean Squared Error between actual and predicted values."""
    return np.sqrt(np.mean((y_actual - y_pred) ** 2))


def power_law(x, a, b):
    """Power law function used in curve fitting."""
    return a * np.power(x, b)


class PowerLawModel:
    def __init__(self, params):
        self.params = params

    def __call__(self, x):
        return power_law(x, *self.params)


@dataclass()
class FitResult:
    """A class used to represent the result of a model fitting."""

    model: np.poly1d or callable
    stage_fit: np.ndarray
    discharge_fit: np.ndarray
    label: str
    rmse: float
    min_discharge: float
    max_discharge: float
    training_start: datetime
    training_end: datetime

    def predict(self, stage, normalized=False):
        """
        Predict discharge based on the given stage.
        If normalized is True, normalize the predicted discharge based on the min and max discharge of the fit.
        """
        discharge = self.model(stage)
        if normalized:
            discharge = (discharge - self.min_discharge) / (
                self.max_discharge - self.min_discharge
            )
        return discharge

    def __eq__(self, other):
        if not other:
            return False
        return (
            self.label == other.label
            and self.training_start == other.training_start
            and self.training_end == other.training_end
        )


def _apply_power_law(val, *args):
    return power_law(val, *args)


@st.cache_data
def fit_power_law(stage, discharge, training_start: datetime, training_end: datetime):
    """Fit a power law model to the given stage and discharge data."""
    params = curve_fit(power_law, stage, discharge, maxfev=5000)[0]
    model = PowerLawModel(params)
    stage_fit = np.linspace(stage.min(), stage.max(), 100)
    discharge_fit = model(stage_fit)
    label = f"Fit: a={params[0]:.3f}, b={params[1]:.3f}"
    rmse = calculate_rmse(discharge, model(stage))
    return FitResult(
        model,
        stage_fit,
        discharge_fit,
        label,
        rmse,
        discharge.min(),
        discharge.max(),
        training_start=training_start,
        training_end=training_end,
    )


@st.cache_data
def fit_polynomial(
    stage, discharge, fit_degree: int, training_start: datetime, training_end: datetime
):
    """Fit a polynomial model of given degree to the stage and discharge data."""
    params = np.polyfit(stage, discharge, fit_degree)
    model = np.poly1d(params)
    stage_fit = np.linspace(stage.min(), stage.max(), 100)
    discharge_fit = model(stage_fit)
    label = f"Fit: {model}"
    rmse = calculate_rmse(discharge, model(stage))
    return FitResult(
        model,
        stage_fit,
        discharge_fit,
        label,
        rmse,
        discharge.min(),
        discharge.max(),
        training_start=training_start,
        training_end=training_end,
    )


DATA_PATH = Path(__file__).parent / "waterdata2008.tsv"


@st.cache_data
def load_water_data():
    """Load water data from CSV file and return as DataFrame with necessary processing and normalization."""

    df = pd.read_csv(DATA_PATH, sep="\t", dtype=str, low_memory=False)
    df["date"] = pd.to_datetime(df["datetime"])
    df["discharge_rate"] = df["148640_00060_00003"].astype(float)
    df["stage_val"] = df["148641_00065_00003"].astype(float)
    # Normalize the stage and discharge values
    df["stage_val_norm"] = (df["stage_val"] - df["stage_val"].min()) / (
        df["stage_val"].max() - df["stage_val"].min()
    )
    df["discharge_rate_norm"] = (df["discharge_rate"] - df["discharge_rate"].min()) / (
        df["discharge_rate"].max() - df["discharge_rate"].min()
    )
    df = df.sort_values(by=["date"])

    return df


# @st.cache_data
# def load_water_data():
#     """Load water data from CSV file and return as DataFrame with necessary processing and normalization."""
#
#     df = pd.read_csv(DATA_PATH, sep="\t", dtype=str, low_memory=False)
#     df = df.groupby(["begin_yr", "month_nu", "day_nu"], as_index=False).aggregate(list)
#     df["date"] = pd.to_datetime(
#         df[["begin_yr", "month_nu", "day_nu"]].rename(
#             columns={"begin_yr": "year", "month_nu": "month", "day_nu": "day"}
#         )
#     )
#     df[["discharge_rate", "stage_val"]] = df["mean_va"].apply(pd.Series)
#     df["discharge_rate"] = df["discharge_rate"].astype(int)
#     df["stage_val"] = df["stage_val"].astype(float)
#     df = df.dropna(subset=["discharge_rate", "stage_val"])
#     # Normalize the stage and discharge values
#     df["stage_val_norm"] = (df["stage_val"] - df["stage_val"].min()) / (df["stage_val"].max() - df["stage_val"].min())
#     df["discharge_rate_norm"] = (df["discharge_rate"] - df["discharge_rate"].min()) / (
#         df["discharge_rate"].max() - df["discharge_rate"].min()
#     )
#
#     df = df.sort_values(by=["date"])
#
#     return df


main()
