import re
from io import BytesIO
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from html import escape

from lens_core import *
from lens_core import (ensure_session_defaults, _confidence_status, _event_values, _finalize_dataset_for_workspace, _get_analysis_areas_cached, _inference_sample, _initial_quality_report, _invalidate_analysis_cache)

# lens_ui.py is executed on every Streamlit rerun. Initialize the current
# session explicitly so fresh users/tabs never depend on lens_core import timing.
ensure_session_defaults()
from lens_dashboards import *
from lens_findings import *

# GLOBAL CSS
# ================================================================

st.html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Sora:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 3%, rgba(110, 72, 223, 0.30), transparent 27%),
        radial-gradient(circle at 90% 15%, rgba(46, 89, 190, 0.15), transparent 30%),
        linear-gradient(135deg, #0b0926 0%, #101434 48%, #172d5b 100%);
    background-attachment: fixed;
}

.missing-data-note {
    margin-top: 10px;
    color: #FF79C6;
    font-size: 0.92rem;
    font-weight: 600;
    line-height: 1.45;
}

.block-container {
    max-width: 1420px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

/* ============================================================
   EVERYTHING BELOW IS THE ORIGINAL LENS WORKSPACE LOOK
   ============================================================ */

.lens-panel {
    padding: 28px;
    margin-bottom: 22px;
    border-radius: 27px;
    background: rgba(255,255,255,0.94);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.35);
    box-shadow: 0 18px 48px rgba(4,10,32,0.19);
}

.lens-title {
    margin-bottom: 6px;
    color: #1D2747;
    font-size: 28px;
    font-weight: 840;
}

.lens-caption {
    color: #777F94;
    font-size: 14px;
    line-height: 1.55;
}

.st-key-upload_panel {
    padding: 28px;
    border-radius: 27px;
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(255,255,255,0.35);
    box-shadow: 0 18px 48px rgba(4,10,32,0.18);
    margin-bottom: 25px;
}

.st-key-upload_panel h3,
.st-key-upload_panel p,
.st-key-upload_panel label {
    color: #1D2747 !important;
}

.dataset-strip {
    display: flex;
    align-items: center;
    gap: 11px;
    margin-bottom: 22px;
    padding: 15px 18px;
    border-radius: 19px;
    background: rgba(21,25,63,0.84);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 14px 36px rgba(4,9,30,0.17);
}

.dataset-dot {
    width: 9px;
    height: 9px;
    flex: 0 0 9px;
    border-radius: 50%;
    background: #45E1DA;
    box-shadow: 0 0 15px rgba(69,225,218,0.70);
}

.dataset-name {
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 780;
}

.dataset-meta {
    margin-top: 2px;
    color: rgba(255,255,255,0.54);
    font-size: 11px;
}

.st-key-mapping_shell {
    padding: 28px;
    border-radius: 27px;
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(255,255,255,0.38);
    box-shadow: 0 18px 48px rgba(4,10,32,0.19);
}

.st-key-mapping_shell h2,
.st-key-mapping_shell h3,
.st-key-mapping_shell p,
.st-key-mapping_shell label {
    color: #1D2747 !important;
}

.mapping-intro {
    padding: 22px;
    margin-bottom: 19px;
    border-radius: 23px;
    background: linear-gradient(135deg, rgba(101,82,222,0.075), rgba(69,225,218,0.07));
    border: 1px solid rgba(89,84,203,0.08);
}

.mapping-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-top: 16px;
}

.mapping-card {
    padding: 19px;
    border-radius: 20px;
    background: #FFFFFF;
    border: 1px solid rgba(31,39,70,0.07);
    box-shadow: 0 8px 24px rgba(31,39,70,0.04);
}

.mapping-label {
    margin-bottom: 7px;
    color: #9198AA;
    font-size: 11px;
    font-weight: 760;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.mapping-value {
    color: #202947;
    font-size: 19px;
    font-weight: 760;
}

.mapping-section-heading {
    color: #202947;
    font-size: 16px;
    font-weight: 820;
    letter-spacing: -.2px;
}

.mapping-section-copy {
    margin-top: 4px;
    color: #7B8398;
    font-size: 12px;
    line-height: 1.5;
}

.mapping-required {
    display: inline-flex;
    margin-top: 12px;
    padding: 5px 8px;
    border-radius: 999px;
    color: #169C58;
    background: rgba(45,212,116,.10);
    border: 1px solid rgba(45,212,116,.18);
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .7px;
    text-transform: uppercase;
}

.optional-mapping-shell {
    margin-top: 18px;
    padding: 22px;
    border-radius: 23px;
    background: linear-gradient(135deg, rgba(22,28,68,.035), rgba(69,225,218,.035));
    border: 1px solid rgba(31,39,70,.07);
}

.optional-mapping-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 18px;
    margin-bottom: 16px;
}

.optional-mapping-count {
    flex-shrink: 0;
    padding: 7px 10px;
    border-radius: 999px;
    background: rgba(69,225,218,.09);
    border: 1px solid rgba(69,225,218,.18);
    color: #169F9A;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .65px;
    text-transform: uppercase;
}

.optional-map-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
}

.optional-map-card {
    min-width: 0;
    padding: 14px 15px;
    border-radius: 15px;
    background: rgba(255,255,255,.84);
    border: 1px solid rgba(31,39,70,.07);
}

.optional-map-card.detected {
    background: linear-gradient(145deg, rgba(69,225,218,.07), rgba(255,255,255,.96) 48%);
    border-color: rgba(69,225,218,.17);
}

.optional-map-card.missing {
    opacity: .72;
    background: rgba(246,247,250,.78);
}

.optional-map-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

.optional-map-label {
    min-width: 0;
    color: #81889C;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .8px;
    text-transform: uppercase;
}

.optional-map-status {
    color: #24AFA8;
    font-size: 8px;
    font-weight: 800;
    letter-spacing: .6px;
    text-transform: uppercase;
}

.optional-map-card.missing .optional-map-status {
    color: #9DA2B0;
}

.optional-map-value {
    margin-top: 7px;
    color: #202947;
    font-size: 13px;
    line-height: 1.35;
    font-weight: 720;
    overflow-wrap: anywhere;
}

.optional-map-card.missing .optional-map-value {
    color: #9AA0AF;
    font-weight: 600;
}

.custom-dimensions-block {
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid rgba(31,39,70,.07);
}

.custom-dimensions-title {
    color: #202947;
    font-size: 12px;
    font-weight: 800;
}

.custom-dimensions-copy {
    margin-top: 4px;
    color: #868DA0;
    font-size: 11px;
}

.custom-dimension-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 11px;
}

.custom-dimension-chip {
    padding: 6px 9px;
    border-radius: 999px;
    color: #6455C8;
    background: rgba(103,83,224,.075);
    border: 1px solid rgba(103,83,224,.12);
    font-size: 9px;
    font-weight: 700;
}

.custom-dimension-empty {
    color: #A1A6B3;
    font-size: 10px;
}

@media (max-width: 1100px) {
    .optional-map-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 700px) {
    .optional-mapping-head { flex-direction: column; }
    .optional-map-grid { grid-template-columns: 1fr; }
}

.st-key-confirm_mapping button {
    border: none !important;
    border-radius: 999px !important;
    background: linear-gradient(135deg, #2DD474, #199D56) !important;
    color: #FFFFFF !important;
    font-weight: 780 !important;
    box-shadow: 0 9px 24px rgba(45,212,116,0.23) !important;
    transition: all .25s ease !important;
}

.st-key-confirm_mapping button:hover {
    transform: translateY(-2px);
    box-shadow: 0 13px 30px rgba(45,212,116,0.34) !important;
}

div[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
    border: none !important;
    border-radius: 999px !important;
    background: linear-gradient(135deg, #2DD474, #199D56) !important;
    color: #FFFFFF !important;
    font-weight: 780 !important;
}

.st-key-workspace_nav {
    max-width: 920px;
    margin: 6px auto 36px auto;
    padding: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(103,83,224,0.13), rgba(69,225,218,0.12));
    border: 1px solid rgba(126,119,230,0.12);
    box-shadow: 0 15px 40px rgba(3,9,30,0.16);
    backdrop-filter: blur(14px);
}

.st-key-workspace_nav div[data-testid="stHorizontalBlock"] {
    gap: 8px !important;
}

.st-key-workspace_nav button {
    min-height: 54px !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    background: transparent !important;
    color: #B7B0FF !important;
    font-size: 14px !important;
    font-weight: 760 !important;
    transition: transform .24s ease, box-shadow .24s ease, background .24s ease, color .24s ease !important;
}

.st-key-workspace_nav button:hover {
    transform: scale(1.055);
    background: rgba(69,225,218,0.10) !important;
    color: #61EFE8 !important;
    border-color: rgba(69,225,218,0.23) !important;
    box-shadow: 0 0 25px rgba(69,225,218,0.17) !important;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 18px;
}

.summary-card {
    padding: 21px;
    border-radius: 22px;
    background: rgba(255,255,255,0.96);
    border: 1px solid rgba(255,255,255,0.34);
    box-shadow: 0 12px 32px rgba(4,10,32,0.14);
    transition: transform .23s ease, box-shadow .23s ease;
}

.summary-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 40px rgba(69,225,218,0.16);
}

.summary-label {
    margin-bottom: 9px;
    color: #878EA2;
    font-size: 11px;
    font-weight: 760;
    letter-spacing: 1.1px;
    text-transform: uppercase;
}

.summary-value {
    color: #202947;
    font-size: 29px;
    font-weight: 850;
    letter-spacing: -1px;
}


/* ============================================================
   ANALYSIS READINESS
   ============================================================ */

.analysis-readiness {
    margin-top: 30px;
    padding-top: 26px;
    border-top: 1px solid rgba(31,39,70,.08);
}

.readiness-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 20px;
    margin-bottom: 16px;
}

.readiness-title {
    color: #202947;
    font-size: 20px;
    font-weight: 820;
}

.readiness-subtitle {
    margin-top: 5px;
    color: #878EA2;
    font-size: 13px;
}

.readiness-score {
    color: #24BDB6;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .7px;
    white-space: nowrap;
}

.analysis-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.analysis-capability {
    position: relative;
    min-height: 128px;
    padding: 17px;
    border-radius: 17px;
    background: #FFFFFF;
    border: 1px solid rgba(31,39,70,.08);
}

.analysis-capability.ready {
    border-color: rgba(69,225,218,.35);
    background:
        linear-gradient(
            145deg,
            rgba(69,225,218,.075),
            rgba(255,255,255,.98) 50%
        );
}

.analysis-capability.limited {
    border-color: rgba(226,176,77,.30);
    background:
        linear-gradient(
            145deg,
            rgba(255,202,94,.08),
            rgba(255,255,255,.98) 50%
        );
}

.analysis-capability.missing {
    background: rgba(246,247,250,.80);
    border-color: rgba(31,39,70,.06);
}

.cap-status {
    display: inline-flex;
    align-items: center;
    margin-bottom: 13px;
    padding: 5px 8px;
    border-radius: 999px;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .8px;
    text-transform: uppercase;
}

.ready .cap-status {
    color: #078D87;
    background: rgba(69,225,218,.13);
}

.limited .cap-status {
    color: #9B7015;
    background: rgba(255,202,94,.18);
}

.missing .cap-status {
    color: #9198A9;
    background: rgba(99,108,130,.08);
}

.cap-title {
    color: #202947;
    font-size: 14px;
    font-weight: 780;
    margin-bottom: 6px;
}

.missing .cap-title {
    color: #7F8799;
}

.cap-desc {
    color: #7C8498;
    font-size: 11px;
    line-height: 1.5;
}

.cap-reason {
    color: #9A8390;
    font-size: 10px;
    line-height: 1.45;
    margin-top: 8px;
}

@media (max-width: 900px) {
    .analysis-grid {
        grid-template-columns: 1fr 1fr;
    }
}

@media (max-width: 600px) {
    .analysis-grid {
        grid-template-columns: 1fr;
    }

    .readiness-header {
        align-items: flex-start;
        flex-direction: column;
    }
}

/* ============================================================
   ANALYSIS WORKSPACE
   ============================================================ */

.analysis-home {
    position: relative;
    overflow: hidden;
    padding: 30px 32px;
    margin-bottom: 18px;
    border-radius: 26px;
    background:
        radial-gradient(circle at 84% 18%, rgba(70,225,218,.17), transparent 28%),
        radial-gradient(circle at 66% 115%, rgba(255,79,163,.12), transparent 34%),
        linear-gradient(135deg, rgba(15,12,49,.96), rgba(23,31,78,.96));
    border: 1px solid rgba(255,255,255,.09);
    box-shadow: 0 18px 48px rgba(4,8,30,.25);
}

.analysis-home::after {
    content: '';
    position: absolute;
    width: 170px;
    height: 170px;
    right: 70px;
    top: -76px;
    border-radius: 50%;
    border: 1px solid rgba(122,244,236,.14);
    box-shadow: 0 0 50px rgba(70,225,218,.08);
}

.analysis-intro {
    position: relative;
    z-index: 2;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
}

.analysis-intro-kicker {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 10px;
    color: #46E1DA;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 1.4px;
    text-transform: uppercase;
}

.analysis-intro-kicker::before {
    content: '';
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #FF4FA3;
    box-shadow: 0 0 14px rgba(255,79,163,.45);
}

.analysis-intro-title {
    color: #FFFFFF;
    font-size: 29px;
    font-weight: 850;
    letter-spacing: -.8px;
}

.analysis-intro-copy {
    margin-top: 8px;
    max-width: 760px;
    color: #AAB3D1;
    font-size: 13px;
    line-height: 1.65;
}

.analysis-rule {
    padding: 8px 12px;
    border-radius: 999px;
    color: #FF7FC2;
    background: rgba(255,79,163,.08);
    border: 1px solid rgba(255,79,163,.18);
    font-size: 9px;
    font-weight: 850;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    white-space: nowrap;
}

.st-key-analysis_area_nav {
    margin-bottom: 22px;
    padding: 7px;
    border-radius: 19px;
    background: rgba(10,13,43,.52);
    border: 1px solid rgba(255,255,255,.07);
    box-shadow: 0 12px 30px rgba(3,8,30,.15);
}

.st-key-analysis_area_nav div[data-testid="stHorizontalBlock"] {
    gap: 7px !important;
}

.st-key-analysis_area_nav button {
    min-height: 54px !important;
    border-radius: 14px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    color: #AAB2CF !important;
    font-size: 12px !important;
    font-weight: 760 !important;
    transition: all .22s ease !important;
}

.st-key-analysis_area_nav button:hover {
    transform: translateY(-1px);
    color: #FFFFFF !important;
    background: rgba(255,255,255,.05) !important;
    border-color: rgba(70,225,218,.18) !important;
    box-shadow: 0 0 22px rgba(70,225,218,.08) !important;
}

.analysis-status-line {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    margin: 2px 0 14px 0;
}

.analysis-status {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 10px;
    border-radius: 999px;
    font-size: 9px;
    font-weight: 850;
    letter-spacing: .9px;
    text-transform: uppercase;
}

.analysis-status.ready {
    color: #4BE8DE;
    background: rgba(70,225,218,.10);
    border: 1px solid rgba(70,225,218,.18);
}

.analysis-status.limited {
    color: #FFD47A;
    background: rgba(255,202,94,.10);
    border: 1px solid rgba(255,202,94,.16);
}

.analysis-status.missing {
    color: #AAB2C7;
    background: rgba(126,135,158,.08);
    border: 1px solid rgba(126,135,158,.12);
}

.analysis-status-hint {
    color: #8E98B8;
    font-size: 11px;
}

.analysis-dashboard-head {
    position: relative;
    overflow: hidden;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    margin: 0 0 18px 0;
    padding: 26px 28px;
    border-radius: 24px;
    background:
        radial-gradient(circle at 90% 12%, rgba(70,225,218,.13), transparent 30%),
        radial-gradient(circle at 78% 110%, rgba(255,79,163,.10), transparent 34%),
        linear-gradient(135deg, rgba(13,15,48,.94), rgba(18,29,71,.94));
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0 15px 40px rgba(4,8,30,.18);
}

.analysis-dashboard-eyebrow {
    margin-bottom: 7px;
    color: #46E1DA;
    font-size: 9px;
    font-weight: 850;
    letter-spacing: 1.3px;
    text-transform: uppercase;
}

.analysis-dashboard-title {
    color: #FFFFFF;
    font-size: 27px;
    font-weight: 850;
    letter-spacing: -.6px;
}

.analysis-dashboard-desc {
    margin-top: 7px;
    max-width: 760px;
    color: #A6AECA;
    font-size: 12px;
    line-height: 1.6;
}

.analysis-dashboard-orbit {
    position: relative;
    width: 96px;
    height: 56px;
    flex: 0 0 96px;
}

.analysis-dashboard-orbit::before,
.analysis-dashboard-orbit::after {
    content: '';
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(70,225,218,.20);
}

.analysis-dashboard-orbit::before {
    width: 74px;
    height: 42px;
    right: 0;
    top: 7px;
    transform: rotate(-12deg);
}

.analysis-dashboard-orbit::after {
    width: 50px;
    height: 50px;
    right: 16px;
    top: 3px;
    border-color: rgba(255,79,163,.16);
}

.analysis-dashboard-orbit span {
    position: absolute;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #46E1DA;
    box-shadow: 0 0 14px rgba(70,225,218,.65);
}

.analysis-dashboard-orbit span:nth-child(1) { right: 4px; top: 23px; }
.analysis-dashboard-orbit span:nth-child(2) { right: 45px; top: 3px; background: #FF4FA3; box-shadow: 0 0 14px rgba(255,79,163,.5); }
.analysis-dashboard-orbit span:nth-child(3) { right: 66px; bottom: 2px; background: #8B5CF6; box-shadow: 0 0 14px rgba(139,92,246,.5); }

.analysis-kpi-card {
    position: relative;
    overflow: hidden;
    min-height: 145px;
    padding: 18px 18px 17px;
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(10,13,43,.94), rgba(19,28,67,.92));
    border: 1px solid rgba(255,255,255,.075);
    box-shadow: 0 13px 30px rgba(2,7,27,.20), inset 0 1px 0 rgba(255,255,255,.025);
}

.analysis-kpi-card::after {
    content: '';
    position: absolute;
    width: 90px;
    height: 90px;
    right: -28px;
    top: -28px;
    border-radius: 50%;
    filter: blur(22px);
    opacity: .28;
}

.analysis-kpi-card.turquoise::after { background: #46E1DA; }
.analysis-kpi-card.magenta::after { background: #FF4FA3; }
.analysis-kpi-card.purple::after { background: #8B5CF6; }
.analysis-kpi-card.blue::after { background: #4F8CFF; }

.analysis-kpi-label {
    position: relative;
    z-index: 2;
    margin-bottom: 12px;
    color: #46E1DA;
    font-size: 9px;
    font-weight: 850;
    letter-spacing: 1.1px;
    text-transform: uppercase;
}

.analysis-kpi-card.magenta .analysis-kpi-label { color: #FF79C6; }
.analysis-kpi-card.purple .analysis-kpi-label { color: #AF8CFF; }
.analysis-kpi-card.blue .analysis-kpi-label { color: #7EACFF; }

.analysis-kpi-value {
    position: relative;
    z-index: 2;
    color: #FFFFFF;
    font-size: clamp(23px, 2.4vw, 37px);
    font-weight: 850;
    line-height: 1.06;
    letter-spacing: -1px;
    overflow-wrap: anywhere;
}

.analysis-kpi-note {
    position: relative;
    z-index: 2;
    margin-top: 11px;
    color: #8993B2;
    font-size: 10px;
    line-height: 1.45;
}

.analysis-chart-heading {
    margin: 10px 0 7px 2px;
}

.analysis-chart-title {
    color: #F4F6FF;
    font-size: 15px;
    font-weight: 800;
}

.analysis-chart-subtitle {
    margin-top: 3px;
    color: #8E97B5;
    font-size: 10px;
    line-height: 1.5;
}

/* Plotly chart shells */
div[data-testid="stPlotlyChart"] {
    overflow: hidden;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,.07);
    background: rgba(8,11,36,.65);
    box-shadow: 0 13px 30px rgba(2,7,27,.16);
}

@media(max-width: 900px) {
    .analysis-intro { align-items: flex-start; flex-direction: column; }
    .analysis-dashboard-orbit { display: none; }
    .analysis-status-line { align-items: flex-start; flex-direction: column; }
}

.st-key-analysis_panel,
.st-key-ask_panel {
    padding: 28px;
    border-radius: 27px;
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(255,255,255,0.35);
    box-shadow: 0 18px 48px rgba(4,10,32,0.19);
}

.st-key-analysis_panel h2,
.st-key-analysis_panel p,
.st-key-ask_panel h2,
.st-key-ask_panel p,
.st-key-ask_panel label {
    color: #1D2747 !important;
}

.st-key-data_fab {
    position: fixed;
    right: 27px;
    bottom: 27px;
    z-index: 9999;
}

.st-key-data_fab button {
    min-height: 49px !important;
    padding: 0 19px !important;
    border-radius: 999px !important;
    border: 1px solid rgba(69,225,218,0.42) !important;
    background: linear-gradient(135deg, #211A5B, #142143) !important;
    color: #FFFFFF !important;
    font-weight: 780 !important;
    letter-spacing: .5px;
    box-shadow: 0 13px 34px rgba(7,9,34,0.28), 0 0 18px rgba(69,225,218,0.12);
    transition: all .25s ease !important;
}

.st-key-data_fab button:hover {
    transform: translateY(-3px) scale(1.035);
    border-color: rgba(69,225,218,0.76) !important;
    box-shadow: 0 17px 40px rgba(7,9,34,0.31), 0 0 27px rgba(69,225,218,0.23);
}

.lens-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    margin-top: 52px;
    padding: 19px 2px 10px;
    border-top: 1px solid rgba(255,255,255,0.13);
    color: rgba(255,255,255,0.52);
    font-size: 12px;
}

.lens-footer strong {
    color: #FF4FA3;
}

@media (max-width: 1000px) {
    .summary-grid { grid-template-columns: 1fr 1fr; }
    .mapping-grid { grid-template-columns: 1fr; }
    .hero-orb-wrap { opacity: .75; right: -7%; }
}

@media (max-width: 700px) {
    .hero-title { font-size: 60px; letter-spacing: -4px; }
    .hero-signature { position: static; display: inline-block; margin-bottom: 18px; }
    .hero-orb-wrap { width: 350px; height: 300px; right: -20%; top: 110px; opacity: .58; }
    .summary-grid { grid-template-columns: 1fr; }
}

@media (prefers-reduced-motion: reduce) {
    .hero-orb-wrap,
    .hero-speck { animation: none !important; }
}
/* LENS editorial hero */
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1280px; padding: 2.5rem 2.5rem 3rem; }
.lens-hero { position:relative; color:#fff; margin:0 0 28px; }
.lens-masthead { display:flex; align-items:center; justify-content:space-between; gap:20px; padding:4px 0 24px; border-bottom:1px solid rgba(255,255,255,.12); }
.lens-brand { display:flex; align-items:center; gap:11px; font:800 23px 'Sora','Inter',sans-serif; letter-spacing:-1px; }
.lens-mark { display:inline-block; width:23px; height:23px; border:2px solid #45E1DA; border-radius:7px; transform:rotate(-12deg); box-shadow:inset 0 0 0 5px #101434; background:rgba(69,225,218,.28); }
.hero-signature { color:#FF4FA3; font-size:10px; font-weight:700; letter-spacing:1.8px; }
.hero-layout { display:grid; grid-template-columns:1.15fr 1fr; align-items:center; gap:32px; padding:46px 0 32px; }
.hero-kicker { color:#B7B0FF; font-size:10px; font-weight:700; letter-spacing:2px; margin-bottom:22px; }
.hero-title { margin:0 0 22px; padding:0; color:#fff !important; font-family:'Sora','Inter',sans-serif; font-size:clamp(44px,5.2vw,70px); line-height:1.09; font-weight:700; letter-spacing:-3.5px; }
.hero-title span { color:#45E1DA; }
.hero-subtitle { max-width:430px; color:#B8BED2; font-size:16px; line-height:1.75; margin:0; }
.hero-pill-row { display:flex; flex-wrap:wrap; gap:10px 20px; margin-top:28px; }
.hero-pill { display:flex; align-items:center; gap:7px; color:#DEE1EE; font-size:11px; font-weight:500; }
.hero-pill::before { content:''; width:4px; height:4px; background:#45E1DA; border-radius:50%; }
.signal-art { position:relative; isolation:isolate; min-width:0; }
.signal-art::before { content:''; position:absolute; z-index:-1; inset:5%; background:radial-gradient(ellipse,rgba(69,225,218,.11),transparent 68%); filter:blur(24px); }
.signal-art svg { display:block; width:100%; height:auto; overflow:visible; }
.hero-art-caption { display:flex; justify-content:space-between; gap:10px; color:#9DA8C4; font-size:9px; letter-spacing:1.8px; padding:0 18px; }
.lens-panel,.st-key-upload_panel,.st-key-mapping_shell,.st-key-analysis_panel,.st-key-ask_panel { border-radius:20px; padding:30px; box-shadow:0 16px 48px rgba(4,10,32,.16); }
.st-key-upload_panel { position:relative; border-top:3px solid #45E1DA; }
.st-key-upload_panel h3 { font-size:24px; letter-spacing:-.7px; padding-top:0; }
.st-key-upload_panel [data-testid="stFileUploaderDropzone"] { border:1px dashed #AAB5C9; border-radius:12px; background:rgba(235,240,248,.65); padding:20px; }
.st-key-upload_panel [data-testid="stCaptionContainer"] p { color:#69738A !important; }
.mapping-card,.summary-card { border-radius:14px; box-shadow:none; }
.summary-value { font-size:clamp(20px,2.25vw,29px); overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }
.mapping-value,.dataset-name { overflow-wrap:anywhere; }
button:focus-visible,a:focus-visible { outline:3px solid #45E1DA !important; outline-offset:3px; }
.lens-footer { margin-top:32px; }
@media(max-width:900px) {
 .hero-layout { gap:12px; grid-template-columns:1.15fr 1fr; }
 .hero-title { font-size:48px; }
 .hero-pill-row { gap:10px; }
}
@media(max-width:640px) {
 .block-container { padding:1.5rem 1rem 5rem; }
 .hero-layout { grid-template-columns:1fr; padding:30px 0 20px; gap:10px; }
 .hero-title { font-size:46px; letter-spacing:-2.5px; }
 .hero-kicker { font-size:9px; letter-spacing:1.3px; }
 .hero-signature { font-size:9px; letter-spacing:1px; }
 .signal-art { max-width:360px; width:100%; margin:0 auto; }
 .lens-panel,.st-key-upload_panel,.st-key-mapping_shell,.st-key-analysis_panel,.st-key-ask_panel { padding:21px; border-radius:16px; }
 .lens-footer { align-items:flex-start; flex-direction:column; gap:8px; }
}
@media(prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none !important; transition:none !important; } }


/* Layered atmosphere and a different hierarchy for each workflow stage. */
.stApp {
 background:
  radial-gradient(ellipse at 3% 8%,rgba(108,60,210,.40),transparent 43%),
  radial-gradient(ellipse at 92% 32%,rgba(36,128,157,.25),transparent 46%),
  radial-gradient(ellipse at 48% 100%,rgba(75,48,150,.24),transparent 55%),
  linear-gradient(130deg,#0b0926 0%,#101434 55%,#172d5b 100%);
 background-attachment:fixed;
}
.block-container { max-width:1360px; padding-top:2rem; }
.lens-hero { isolation:isolate; overflow:hidden; padding:0; min-height:0; border:1px solid rgba(183,176,255,.17); border-radius:26px; background:rgba(12,13,40,.38); box-shadow:0 24px 90px rgba(3,5,24,.18),inset 0 1px 0 rgba(255,255,255,.06); }
.hero-atmosphere { position:absolute; inset:0; pointer-events:none; z-index:-1; background:radial-gradient(ellipse at 86% 26%,rgba(69,225,218,.19),transparent 42%),radial-gradient(ellipse at 46% 112%,rgba(129,79,237,.28),transparent 57%),linear-gradient(112deg,transparent 48%,rgba(183,176,255,.045) 48.1%,transparent 70%); }
.hero-atmosphere::after { content:''; position:absolute; inset:0; background-image:linear-gradient(rgba(183,176,255,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(183,176,255,.045) 1px,transparent 1px); background-size:48px 48px; mask-image:linear-gradient(90deg,transparent 25%,#000 100%); }
.lens-masthead { position:relative; z-index:2; margin:0 34px; padding:22px 0; }
.masthead-label { display:flex; align-items:center; gap:10px; color:#C2C3DC; font-size:9px; font-weight:650; letter-spacing:1.6px; }
.brand-indicator { width:6px; height:6px; flex-shrink:0; border-radius:50%; background:#45E1DA; box-shadow:0 0 14px rgba(69,225,218,.6); }
.hero-signature { margin:0; }
.lens-welcome .hero-layout { grid-template-columns:1.12fr 1fr; gap:8px; padding:32px 40px 34px; }
.hero-content { position:relative; z-index:2; }
.lens-welcome .hero-kicker { margin:0 0 14px; font-size:9px; letter-spacing:1.9px; }
.hero-wordmark { margin:0 0 18px !important; padding:0 !important; font-family:'Sora','Inter',sans-serif; font-size:clamp(90px,10.7vw,146px) !important; line-height:.95 !important; font-weight:800 !important; letter-spacing:-9px !important; color:#fff !important; text-shadow:0 8px 45px rgba(183,176,255,.13); }
.wordmark-dot { color:#45E1DA; }
.welcome-line { margin:0 0 16px !important; padding:0 !important; color:#fff !important; font-family:'Inter',sans-serif; font-size:clamp(24px,2.4vw,32px) !important; line-height:1.24 !important; font-weight:600 !important; letter-spacing:-1px !important; }
.welcome-line span { color:#B7B0FF; }
.lens-welcome .hero-subtitle { max-width:370px; font-size:14px; line-height:1.7; }
.lens-welcome .signal-art { align-self:stretch; display:flex; flex-direction:column; justify-content:center; min-height:300px; }
.signal-art svg { position:relative; z-index:1; filter:drop-shadow(0 18px 24px rgba(0,0,0,.2)); }
.optical-ring { position:absolute; width:290px; height:290px; left:50%; top:48%; border:1px solid rgba(69,225,218,.2); border-radius:50%; transform:translate(-50%,-50%) rotate(-25deg); pointer-events:none; }
.ring-one { width:360px; height:285px; border-color:rgba(183,176,255,.19); box-shadow:0 0 55px rgba(69,225,218,.06),inset 0 0 55px rgba(69,225,218,.035); }
.ring-two { width:410px; height:230px; transform:translate(-50%,-50%) rotate(-48deg); border-color:rgba(69,225,218,.18); }
.ring-three { width:220px; height:370px; transform:translate(-50%,-50%) rotate(-34deg); border-color:rgba(183,176,255,.13); }
.ring-one::after { content:''; position:absolute; left:38px; top:29px; width:6px; height:6px; background:#45E1DA; border-radius:50%; box-shadow:0 0 16px #45E1DA; }
.art-coordinate { position:absolute; top:4px; right:4px; color:#9DA8C4; font-size:8px; letter-spacing:1.6px; }
.hero-art-caption { position:relative; z-index:2; margin-top:8px; font-size:8px; }
.hero-capabilities { display:grid; grid-template-columns:repeat(3,1fr); gap:0; border-top:1px solid rgba(183,176,255,.15); background:rgba(183,176,255,.035); }
.capability { padding:20px 28px; display:flex; align-items:center; gap:14px; color:#E0E3F1; font-size:12px; font-weight:550; }
.capability + .capability { border-left:1px solid rgba(183,176,255,.15); }
.cap-number { color:#9997BF; font-size:10px; font-variant-numeric:tabular-nums; }
.cap-glyph { margin-left:auto; color:#45E1DA; font-size:17px; }
.lens-compact { margin-bottom:24px; }
.lens-compact .lens-masthead { padding:17px 0; }
.lens-compact .lens-brand { font-size:25px; }
.compact-content { position:relative; z-index:2; padding:26px 34px 30px; max-width:790px; }
.compact-content .hero-kicker { margin:0 0 10px; font-size:9px; }
.compact-title { margin:0 0 12px !important; padding:0 !important; font-family:'Sora','Inter',sans-serif; font-size:clamp(28px,3.3vw,43px) !important; font-weight:650 !important; color:#fff !important; line-height:1.18 !important; letter-spacing:-1.8px !important; }
.compact-content .hero-subtitle { max-width:660px; font-size:13px; }
.compact-orbits { position:absolute; right:-30px; top:60px; width:240px; height:240px; border:1px solid rgba(69,225,218,.16); border-radius:50%; box-shadow:0 0 0 32px rgba(69,225,218,.025),0 0 0 65px rgba(183,176,255,.025); pointer-events:none; }
@media(max-width:1000px) {
 .lens-welcome .hero-layout { padding:32px 28px; }
 .hero-wordmark { font-size:100px !important; letter-spacing:-7px !important; }
 .optical-ring { opacity:.7; }
 .ring-one { width:290px; height:245px; }.ring-two { width:325px; height:200px; }.ring-three { width:190px; height:300px; }
 .capability { padding:18px; gap:9px; font-size:11px; }
}
@media(max-width:700px) {
 .block-container { padding:1.5rem 1rem 5rem; }
 .lens-masthead { margin:0 22px; gap:12px; padding:18px 0; }
 .masthead-label { max-width:210px; font-size:8px; line-height:1.6; letter-spacing:1px; }
 .hero-signature { white-space:nowrap; font-size:8px; letter-spacing:.8px; }
 .lens-welcome .hero-layout { grid-template-columns:1fr; padding:28px 22px 22px; }
 .hero-wordmark { font-size:clamp(76px,20vw,120px) !important; letter-spacing:-6px !important; }
 .lens-welcome .signal-art { width:100%; min-height:210px; max-width:340px; margin:20px auto 0; }
 .art-coordinate { top:-5px; }
 .hero-capabilities { grid-template-columns:1fr; }
 .capability { padding:13px 22px; }
 .capability + .capability { border-left:0; border-top:1px solid rgba(183,176,255,.1); }
 .lens-hero { border-radius:20px; min-height:0; }
 .compact-content { padding:24px 22px; }
 .compact-title { letter-spacing:-1px !important; }
 .compact-orbits { opacity:.35; }
}

/* ============================================================
   AUTOMATIC FINDINGS
   ============================================================ */
.findings-shell { position:relative; overflow:hidden; margin:0 0 28px; padding:30px; border-radius:28px; border:1px solid rgba(183,176,255,.14); background:radial-gradient(circle at 92% 8%,rgba(69,225,218,.15),transparent 28%),radial-gradient(circle at 12% 110%,rgba(255,79,163,.13),transparent 36%),linear-gradient(135deg,rgba(10,11,40,.97),rgba(18,25,68,.97)); box-shadow:0 22px 60px rgba(3,6,25,.25); }
.findings-head { display:flex; align-items:flex-start; justify-content:space-between; gap:28px; }
.findings-kicker { color:#46E1DA; font-size:9px; font-weight:850; letter-spacing:1.7px; margin-bottom:9px; }
.findings-title { color:#fff; font-family:'Sora','Inter',sans-serif; font-size:32px; font-weight:700; letter-spacing:-1.2px; }
.findings-copy { max-width:760px; margin-top:8px; color:#AEB7D3; font-size:13px; line-height:1.65; }
.findings-scorebox { min-width:118px; padding:15px 18px; text-align:right; border:1px solid rgba(69,225,218,.18); border-radius:18px; background:rgba(69,225,218,.055); }
.findings-score { color:#fff; font-size:30px; font-weight:850; line-height:1; }
.findings-score-label { margin-top:4px; color:#46E1DA; font-size:9px; font-weight:800; letter-spacing:1px; text-transform:uppercase; }
.findings-score-meta { margin-top:8px; color:#8994B5; font-size:9px; }
.findings-path { display:flex; align-items:center; gap:12px; margin:22px 0 18px; padding:13px 16px; border-radius:15px; border:1px solid rgba(255,255,255,.07); background:rgba(255,255,255,.035); }
.findings-path span { color:#8994B5; font-size:9px; font-weight:800; letter-spacing:1px; text-transform:uppercase; }
.findings-path strong { color:#F2F4FF; font-size:12px; font-weight:650; }
.findings-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
.finding-card { position:relative; overflow:hidden; min-height:232px; padding:20px; border-radius:20px; border:1px solid rgba(255,255,255,.08); background:rgba(255,255,255,.035); }
.finding-card::before { content:''; position:absolute; left:0; top:0; bottom:0; width:3px; background:#66708F; }
.finding-card.priority-high::before { background:linear-gradient(#FF4FA3,#8B5CF6); }
.finding-card.priority-medium::before { background:linear-gradient(#B7B0FF,#6D5DFB); }
.finding-card.priority-watch::before { background:linear-gradient(#46E1DA,#4F8CFF); }
.finding-card.priority-context::before { background:#65708F; }
.finding-topline { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:13px; }
.finding-priority { padding:5px 8px; border-radius:999px; font-size:8px; font-weight:850; letter-spacing:1px; text-transform:uppercase; color:#fff; background:rgba(255,255,255,.08); }
.priority-high .finding-priority { color:#FF78C2; background:rgba(255,79,163,.10); }
.priority-medium .finding-priority { color:#BEB4FF; background:rgba(139,92,246,.11); }
.priority-watch .finding-priority { color:#69EEE7; background:rgba(69,225,218,.10); }
.finding-category { color:#7884A8; font-size:9px; font-weight:700; }
.finding-title { color:#fff; font-size:18px; font-weight:760; letter-spacing:-.3px; margin-bottom:8px; }
.finding-summary { color:#B9C1D9; font-size:12px; line-height:1.6; min-height:57px; }
.finding-evidence { display:flex; flex-wrap:wrap; gap:6px; margin:13px 0; }
.finding-evidence-chip { padding:6px 8px; border-radius:9px; color:#DDE4F7; font-size:9px; font-weight:650; border:1px solid rgba(69,225,218,.11); background:rgba(69,225,218,.055); }
.finding-next { color:#DDE3F4; font-size:10px; line-height:1.55; margin-top:9px; }
.finding-next b { color:#46E1DA; font-weight:750; }
.finding-note { margin-top:9px; padding-top:9px; border-top:1px solid rgba(255,255,255,.06); color:#7782A3; font-size:9px; line-height:1.5; }
.findings-method-note { margin-top:16px; color:#7E89A8; font-size:9px; line-height:1.55; }
.findings-method-note strong { color:#B7B0FF; }

/* Clickable finding cards — original typography, full-card hit area */
.finding-card-link {
    display:block;
    color:inherit !important;
    text-decoration:none !important;
    border-radius:20px;
}
.finding-card-link:hover { text-decoration:none !important; }
.finding-card-link .finding-card {
    height:100%;
    transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease,background .22s ease;
}
.finding-card-link:hover .finding-card {
    transform:translateY(-3px);
    border-color:rgba(69,225,218,.28);
    background:linear-gradient(135deg,rgba(255,255,255,.055),rgba(69,225,218,.04));
    box-shadow:0 18px 40px rgba(3,7,28,.20),0 0 22px rgba(69,225,218,.07);
}
.finding-card-link:focus-visible .finding-card {
    outline:2px solid rgba(69,225,218,.65);
    outline-offset:3px;
}
.finding-open {
    margin-top:13px;
    color:#8BF4EF;
    font-size:9px;
    font-weight:780;
    letter-spacing:.25px;
}
.finding-empty-card { margin-bottom:14px; }

/* Full-card native Streamlit click target; visible typography stays in the HTML card underneath. */
[class*="st-key-finding_click_"] {
    display:grid !important;
    grid-template-columns:1fr !important;
    margin-bottom:14px;
    position:relative;
}
[class*="st-key-finding_click_"] > div {
    grid-column:1 !important;
    grid-row:1 !important;
}
[class*="st-key-finding_click_"] div[data-testid="stButton"] {
    z-index:50 !important;
    width:100% !important;
    height:100% !important;
    align-self:stretch !important;
    margin:0 !important;
}
[class*="st-key-finding_click_"] div[data-testid="stButton"] > button {
    width:100% !important;
    height:100% !important;
    min-height:232px !important;
    opacity:0.001 !important;
    cursor:pointer !important;
    border-radius:20px !important;
    padding:0 !important;
    margin:0 !important;
}
[class*="st-key-finding_click_"] .finding-card {
    z-index:10;
    transition:transform .20s ease,border-color .20s ease,box-shadow .20s ease;
}
[class*="st-key-finding_click_"]:hover .finding-card {
    transform:translateY(-3px);
    border-color:rgba(69,225,218,.30);
    box-shadow:0 16px 38px rgba(5,9,32,.28),0 0 24px rgba(69,225,218,.08);
}
[class*="st-key-finding_click_"]:hover .finding-card { transform:translateY(-3px); border-color:rgba(69,225,218,.30); box-shadow:0 16px 36px rgba(3,8,31,.22),0 0 24px rgba(69,225,218,.08); }

.evidence-scroll-anchor { scroll-margin-top:34px; }
.analysis-explore-heading { margin:28px 0 14px; }
.analysis-explore-kicker { color:#46E1DA; font-size:9px; font-weight:850; letter-spacing:1.5px; }
.analysis-explore-title { margin-top:5px; color:#fff; font-size:22px; font-weight:760; }
@media(max-width:850px) { .findings-grid{grid-template-columns:1fr;} .findings-head{flex-direction:column;} .findings-scorebox{text-align:left;} .findings-path{align-items:flex-start;flex-direction:column;} }

</style>
"""
)


# ================================================================
# HERO
# ================================================================

# The welcome page leads with the brand; subsequent steps lead with the task.
if st.session_state.workflow_step == "upload":
    st.html(
        """
<section class="lens-hero lens-welcome">
  <div class="hero-atmosphere" aria-hidden="true"></div>
  <div class="lens-masthead">
    <div class="masthead-label"><span class="brand-indicator"></span>PRODUCT ANALYTICS INTELLIGENCE LAYER</div>
    <div class="hero-signature">BY SENA ESER</div>
  </div>
  <div class="hero-layout">
    <div class="hero-content">
      <div class="hero-kicker">A CLEARER WAY TO SEE YOUR PRODUCT</div>
      <h1 class="hero-wordmark">LENS<span class="wordmark-dot">.</span></h1>
      <h2 class="welcome-line">Find the signal.<br><span>See what comes next.</span></h2>
      <p class="hero-subtitle">Turn raw product data into clarity, signals and next decisions.</p>
    </div>
    <div class="signal-art" aria-hidden="true"><div class="optical-ring ring-one"></div><div class="optical-ring ring-two"></div><div class="optical-ring ring-three"></div><span class="art-coordinate">L / 01 — SIGNAL FIELD</span>
      <svg viewBox="0 0 480 300" fill="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="lensGlass" x1="179" y1="50" x2="350" y2="254" gradientUnits="userSpaceOnUse"><stop stop-color="#45E1DA" stop-opacity=".18"/><stop offset="1" stop-color="#B7B0FF" stop-opacity=".025"/></linearGradient>
          <linearGradient id="lensEdge"><stop stop-color="#B7B0FF" stop-opacity=".25"/><stop offset=".5" stop-color="#45E1DA" stop-opacity=".85"/><stop offset="1" stop-color="#45E1DA" stop-opacity=".15"/></linearGradient>
        </defs>
        <path d="M12 80H464M12 150H464M12 220H464M80 32V274M160 32V274M240 32V274M320 32V274M400 32V274" stroke="#B7B0FF" stroke-opacity=".055"/>
        <g stroke="#B7B0FF" stroke-opacity=".22" stroke-dasharray="3 6">
          <path d="M30 80C105 80 115 110 190 110"/><path d="M12 150H190"/><path d="M30 220C110 220 130 190 190 190"/>
        </g>
        <g fill="#B7B0FF"><circle cx="30" cy="80" r="4" opacity=".5"/><circle cx="71" cy="126" r="3" opacity=".4"/><circle cx="42" cy="180" r="4" opacity=".3"/><circle cx="107" cy="93" r="3" opacity=".7"/><circle cx="30" cy="220" r="3" opacity=".65"/><circle cx="124" cy="209" r="4" opacity=".35"/><circle cx="106" cy="150" r="4" opacity=".6"/></g>
        <rect x="160" y="57" width="142" height="193" rx="18" transform="rotate(-12 160 57)" fill="#15193F" fill-opacity=".6" stroke="#B7B0FF" stroke-opacity=".22"/>
        <rect x="180" y="46" width="142" height="208" rx="18" transform="rotate(5 180 46)" fill="#142143" stroke="#B7B0FF" stroke-opacity=".3"/>
        <rect x="190" y="50" width="142" height="204" rx="16" fill="url(#lensGlass)" stroke="url(#lensEdge)"/>
        <path d="M210 80H236M210 86H223" stroke="#B7B0FF" stroke-opacity=".5" stroke-width="2" stroke-linecap="round"/>
        <path d="M216 169L243 142L267 153L304 113" stroke="#45E1DA" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="304" cy="113" r="6" fill="#45E1DA"/><circle cx="304" cy="113" r="13" stroke="#45E1DA" stroke-opacity=".25"/>
        <path d="M212 212H309M212 222H270" stroke="#B7B0FF" stroke-opacity=".25" stroke-width="3" stroke-linecap="round"/>
        <path d="M333 151H425" stroke="#45E1DA" stroke-opacity=".55"/>
        <rect x="389" y="132" width="65" height="38" rx="10" fill="#142143" stroke="#45E1DA" stroke-opacity=".45"/>
        <path d="M413 151H431M425 145L431 151L425 157" stroke="#45E1DA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <div class="hero-art-caption"><span>RAW DATA</span><span>A CLEARER PERSPECTIVE</span></div>
    </div>
  </div>
  <div class="hero-capabilities">
    <div class="capability"><span class="cap-number">01</span><span>Detect structure</span><span class="cap-glyph" aria-hidden="true">↗</span></div>
    <div class="capability"><span class="cap-number">02</span><span>Surface problems</span><span class="cap-glyph" aria-hidden="true">↗</span></div>
    <div class="capability"><span class="cap-number">03</span><span>Generate hypotheses</span><span class="cap-glyph" aria-hidden="true">↗</span></div>
  </div>
</section>
"""
    )
else:
    if st.session_state.workflow_step == "mapping":
        hero_eyebrow = "01 / DATA STRUCTURE"
        hero_heading = "Fine-tune your data." if st.session_state.mapping_edit_mode else "Give your data structure."
        hero_description = "Review the suggested fields, edit the mapping and confirm when everything looks right."
    else:
        hero_eyebrow, hero_heading, hero_description = {
            "summary": ("02 / YOUR WORKSPACE", "Your data, at a glance.", "Start with the big picture. Explore the events, users and time period in your dataset."),
            "analysis": ("02 / ANALYSIS", "Look closer. See more.", "Explore product behavior, friction, patterns and changes over time."),
            "ask_lens": ("02 / ASK LENS", "Start with a better question.", "Bring your product questions into focus with the data you have."),
        }.get(st.session_state.workspace_page, ("02 / YOUR WORKSPACE", "Your data, at a glance.", "Explore your product dataset."))
    st.html(
        f"""
<section class="lens-hero lens-compact">
  <div class="hero-atmosphere" aria-hidden="true"></div>
  <div class="lens-masthead">
    <div class="lens-brand"><span class="lens-mark" aria-hidden="true"></span>LENS</div>
    <div class="hero-signature">BY SENA ESER</div>
  </div>
  <div class="compact-content">
    <div class="hero-kicker">{hero_eyebrow}</div>
    <h1 class="compact-title">{hero_heading}</h1>
    <p class="hero-subtitle">{hero_description}</p>
  </div>
  <div class="compact-orbits" aria-hidden="true"></div>
</section>
"""
    )


# ================================================================
# WORKSPACE NAVIGATION
# ================================================================

def workspace_navigation():
    current_page = st.session_state.workspace_page

    st.html(
        f"""
<style>
.st-key-nav_{current_page} button {{
    background: linear-gradient(135deg, #49E3DC, #2DBDB7) !important;
    color: #08252B !important;
    border-color: rgba(69,225,218,0.55) !important;
    box-shadow: 0 11px 28px rgba(69,225,218,0.30), 0 0 24px rgba(69,225,218,0.18) !important;
    transform: scale(1.035);
}}

.st-key-nav_{current_page} button:hover {{
    transform: scale(1.075);
    box-shadow: 0 14px 34px rgba(69,225,218,0.39), 0 0 32px rgba(69,225,218,0.26) !important;
}}
</style>
"""
    )

    with st.container(key="workspace_nav"):
        nav1, nav2, nav3 = st.columns(3)

        with nav1:
            with st.container(key="nav_summary"):
                if st.button("Summary", key="summary_nav", use_container_width=True):
                    st.session_state.workspace_page = "summary"
                    st.rerun()

        with nav2:
            with st.container(key="nav_analysis"):
                if st.button("Analysis", key="analysis_nav", use_container_width=True):
                    st.session_state.workspace_page = "analysis"
                    st.rerun()

        with nav3:
            with st.container(key="nav_ask_lens"):
                if st.button("Ask LENS", key="ask_lens_nav", use_container_width=True):
                    st.session_state.workspace_page = "ask_lens"
                    st.rerun()


# ================================================================
# FLOATING DATA MENU
# ================================================================

def floating_data_controls():
    with st.container(key="data_fab"):
        with st.popover("✦ DATA"):
            st.markdown("**Data controls**")
            st.caption(st.session_state.dataset_name)

            if st.button("Edit mapping", key="fab_edit_mapping", use_container_width=True):
                st.session_state.workflow_step = "mapping"
                st.session_state.mapping_edit_mode = True
                st.rerun()

            if st.button("Change dataset", key="fab_change_dataset", use_container_width=True):
                st.session_state.confirm_dataset_change = True

            if st.session_state.confirm_dataset_change:
                st.warning("Changing the dataset will reset the current analysis session.")

                if st.button("Yes, change dataset", key="fab_confirm_change", use_container_width=True):
                    reset_dataset()
                    st.rerun()

                if st.button("Cancel", key="fab_cancel_change", use_container_width=True):
                    st.session_state.confirm_dataset_change = False
                    st.rerun()


@st.cache_data(show_spinner=False, max_entries=4)
def _prepare_uploaded_csv(raw_bytes):
    """Parse once; infer structure from a bounded sample for fast first load."""
    df = pd.read_csv(BytesIO(raw_bytes), low_memory=False)
    df = sanitize_duplicate_columns(df)
    inference_df = _inference_sample(df)
    profile = profile_dataset(inference_df)
    profile["rows"] = int(len(df))
    profile["profile_sample_rows"] = int(len(inference_df))
    profile["duplicate_rows"] = None
    schema_results = detect_schema(inference_df)
    grain_info = detect_data_grain(inference_df, schema_results=schema_results, profile=profile)
    return df, inference_df, profile, schema_results, grain_info


# ================================================================
# UPLOAD
# ================================================================

# Public-demo upload guardrails. LENS deliberately caps uploads below
# Streamlit's platform maximum so one large CSV cannot exhaust a free
# Community Cloud instance for everyone else.
MAX_UPLOAD_MB = 85
MAX_UPLOAD_ROWS = 1_000_000


def _render_upload_limit_note():
    st.caption(
        f"Public demo limit: up to {MAX_UPLOAD_MB} MB and {MAX_UPLOAD_ROWS:,} rows per CSV. "
        "For larger datasets, upload a representative sample."
    )


if st.session_state.workflow_step == "upload":
    with st.container(key="upload_panel"):
        st.markdown("### Start with your product data")
        st.caption("Upload a CSV, or explore LENS instantly with the built-in SaaS demo dataset.")
        _render_upload_limit_note()

        uploaded_file = st.file_uploader(
            "Upload dataset",
            type=["csv"],
            key="dataset_uploader",
        )

        st.markdown(
            '<div style="text-align:center; opacity:.58; font-size:.82rem; margin:.15rem 0 .45rem;">or</div>',
            unsafe_allow_html=True,
        )

        use_sample = st.button(
            "✦ Use sample dataset",
            key="use_sample_dataset",
            use_container_width=True,
        )
        st.caption("Built-in SaaS demo · 12,000 users · 232,949 events · ~20 MB")

    # Both paths intentionally feed the exact same profiling / mapping pipeline.
    # This keeps the demo representative of what a real uploaded CSV experiences.
    raw_bytes = None
    selected_name = None

    if use_sample:
        # Support both project layouts:
        #   PULSE/src/lens_ui.py + PULSE/data/lens_demo_events.csv
        # and a flat extracted package:
        #   src/lens_ui.py + src/data/lens_demo_events.csv
        module_dir = Path(__file__).resolve().parent
        sample_candidates = [
            module_dir / "data" / "lens_demo_events.csv",
            module_dir.parent / "data" / "lens_demo_events.csv",
        ]
        sample_path = next((p for p in sample_candidates if p.exists()), None)

        if sample_path is None:
            st.error(
                "The built-in sample dataset could not be found. Expected "
                "`data/lens_demo_events.csv` in the app folder or project root."
            )
            st.stop()

        with st.spinner("Loading the LENS sample dataset…"):
            raw_bytes = sample_path.read_bytes()
        selected_name = "lens_demo_events.csv"

    elif uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        selected_name = uploaded_file.name

    if raw_bytes is not None:
        upload_size_mb = len(raw_bytes) / (1024 * 1024)
        st.session_state.upload_size_mb = upload_size_mb

        # Reject oversize files before pandas parses them. This keeps the app on
        # the Upload page and avoids a large memory spike on free hosting.
        if upload_size_mb > MAX_UPLOAD_MB:
            st.error(
                f"This CSV is {upload_size_mb:.1f} MB. The public demo accepts files up to "
                f"{MAX_UPLOAD_MB} MB. Please upload a smaller representative sample."
            )
            st.stop()

        with st.spinner("Understanding your dataset…"):
            df, inference_df, profile, schema_results, grain_info = _prepare_uploaded_csv(raw_bytes)

        # A narrow CSV can still contain a very large number of rows, so keep a
        # second guard after parsing. The workflow does not advance when rejected.
        if len(df) > MAX_UPLOAD_ROWS:
            st.error(
                f"This CSV contains {len(df):,} rows. The public demo accepts up to "
                f"{MAX_UPLOAD_ROWS:,} rows. Please upload a smaller representative sample."
            )
            st.stop()

        st.session_state.dataset_df = df
        st.session_state.dataset_name = selected_name
        _invalidate_analysis_cache()
        st.session_state.data_profile = profile
        st.session_state.grain_info = grain_info
        st.session_state.inference_sample_rows = len(inference_df)
        st.session_state.data_quality_finalized = False
        st.session_state.timestamp_invalid_count = None
        st.session_state.user_guess = schema_results["user_id"]
        st.session_state.event_guess = schema_results["event_name"]
        st.session_state.time_guess = schema_results["event_timestamp"]

        # Conservative core assignment. Low-confidence roles remain reviewable instead of being silently trusted.
        st.session_state.user_col = schema_results["user_id"]["best_match"] if schema_results["user_id"]["confidence"] >= 50 else None
        inferred_grain = grain_info["grain"]
        if inferred_grain == "event":
            st.session_state.event_col = schema_results["event_name"]["best_match"] if schema_results["event_name"]["confidence"] >= 55 else None
            st.session_state.timestamp_col = schema_results["event_timestamp"]["best_match"] if schema_results["event_timestamp"]["confidence"] >= 55 else None
        else:
            st.session_state.event_col = None
            st.session_state.timestamp_col = schema_results["event_timestamp"]["best_match"] if schema_results["event_timestamp"]["confidence"] >= 55 else None

        st.session_state.optional_mappings = detect_optional_mappings(inference_df)
        st.session_state.custom_dimensions = detect_custom_dimensions(inference_df, st.session_state.optional_mappings)
        if inferred_grain == "event" and st.session_state.event_col:
            st.session_state.event_role_mappings, st.session_state.event_role_confidence = detect_event_roles(inference_df)
        else:
            st.session_state.event_role_mappings, st.session_state.event_role_confidence = {}, {}
        st.session_state.data_quality = _initial_quality_report()
        st.session_state.semantic_model = build_semantic_model(df)
        st.session_state.optional_mapping_edit_mode = False
        st.session_state.mapping_edit_mode = False
        st.session_state.workflow_step = "mapping"
        st.rerun()


# ================================================================
# MAPPING
# ================================================================

elif st.session_state.workflow_step == "mapping":
    df = st.session_state.dataset_df
    grain_info = st.session_state.get("grain_info") or {"grain": "ambiguous", "confidence": 0, "evidence": []}
    grain = grain_info.get("grain", "ambiguous")

    st.html(f"""
<div class="dataset-strip"><div class="dataset-dot"></div><div>
<div class="dataset-name">{escape(str(st.session_state.dataset_name))}</div>
<div class="dataset-meta">{len(df):,} rows loaded · {escape(grain.title())} data</div>
</div></div>""")


    if not st.session_state.optional_mappings:
        inference_df = _inference_sample(df)
        st.session_state.optional_mappings = detect_optional_mappings(inference_df)
        st.session_state.custom_dimensions = detect_custom_dimensions(inference_df, st.session_state.optional_mappings)
    if grain == "event" and st.session_state.event_col and not st.session_state.event_role_mappings:
        st.session_state.event_role_mappings, st.session_state.event_role_confidence = detect_event_roles(_inference_sample(df))

    optional_cards = ""
    for role, label in OPTIONAL_ROLE_LABELS.items():
        mapped = st.session_state.optional_mappings.get(role)
        status_class = "detected" if mapped else "missing"
        optional_cards += f"""<div class="optional-map-card {status_class}"><div class="optional-map-topline"><div class="optional-map-label">{escape(label)}</div><div class="optional-map-status">{'Detected' if mapped else 'Optional'}</div></div><div class="optional-map-value">{escape(str(mapped)) if mapped else 'Not detected'}</div></div>"""

    event_role_cards = ""
    if grain == "event":
        for role, label in EVENT_ROLE_LABELS.items():
            mapped = st.session_state.event_role_mappings.get(role)
            conf = st.session_state.event_role_confidence.get(role, "Not detected")
            status_class = "detected" if mapped else "missing"
            event_role_cards += f"""<div class="optional-map-card {status_class}"><div class="optional-map-topline"><div class="optional-map-label">{escape(label)}</div><div class="optional-map-status">{escape(conf)}</div></div><div class="optional-map-value">{escape(str(mapped)) if mapped else 'Not detected'}</div></div>"""

    custom_dims = st.session_state.custom_dimensions or []
    custom_dim_html = "".join(f'<span class="custom-dimension-chip">{escape(str(col))}</span>' for col in custom_dims) or '<span class="custom-dimension-empty">No additional categorical dimensions detected.</span>'

    core_cards = []
    user_guess = st.session_state.get("user_guess") or {}
    time_guess = st.session_state.get("time_guess") or {}
    event_guess = st.session_state.get("event_guess") or {}
    core_cards.append(("User identifier", st.session_state.get("user_col"), _confidence_status(user_guess.get("confidence"))))
    if grain == "event":
        core_cards.append(("Event", st.session_state.get("event_col"), _confidence_status(event_guess.get("confidence"))))
    if st.session_state.get("timestamp_col"):
        core_cards.append(("Event / record time", st.session_state.get("timestamp_col"), _confidence_status(time_guess.get("confidence"))))
    core_html = "".join(f'<div class="mapping-card"><div class="mapping-label">{escape(label)}</div><div class="mapping-value">{escape(str(value)) if value else "Not confidently detected"}</div><div class="mapping-required">{escape(status)}</div></div>' for label, value, status in core_cards)

    quality = st.session_state.get("data_quality") or _initial_quality_report()
    quality_copy = ("Full quality scan runs once after structure confirmation." if quality.get("pending_full_scan") else f"{len(quality['issues'])} data-quality item(s) detected. Details remain available in the workspace.")
    evidence_html = "".join(f'<span class="custom-dimension-chip">{escape(str(x))}</span>' for x in grain_info.get("evidence", [])[:4])

    with st.container(key="mapping_shell"):
        st.markdown("## Data Structure")
        st.caption("LENS identifies the table type, then maps only the roles relevant to it. Review anything that looks wrong.")
        st.html(f"""
<div class="mapping-intro">
<div class="mapping-section-heading">Detected grain · {escape(grain.title())}</div>
<div class="mapping-section-copy">LENS will only unlock analyses supported by the detected table structure.</div>
<div class="custom-dimension-row">{evidence_html}</div>
<div class="mapping-grid">{core_html}</div></div>
<div class="optional-mapping-shell"><div class="optional-mapping-head"><div><div class="mapping-section-heading">Detected analytical fields</div><div class="mapping-section-copy">Semantic roles used to unlock only compatible analyses.</div></div><div class="optional-mapping-count">{sum(bool(v) for v in st.session_state.optional_mappings.values())} / {len(OPTIONAL_ROLE_LABELS)} detected</div></div><div class="optional-map-grid">{optional_cards}</div><div class="custom-dimensions-block"><div class="custom-dimensions-title">Custom dimensions</div><div class="custom-dimensions-copy">Other low-cardinality fields usable for segmentation.</div><div class="custom-dimension-row">{custom_dim_html}</div></div></div>
{f'<div class="optional-mapping-shell"><div class="mapping-section-heading">Detected event structure</div><div class="mapping-section-copy">Canonical roles power ordered funnel and lifecycle metrics.</div><div class="optional-map-grid">{event_role_cards}</div></div>' if grain == 'event' else ''}
<div class="optional-mapping-shell"><div class="mapping-section-heading">Data quality</div><div class="mapping-section-copy">{quality_copy}</div></div>
""")

        if not st.session_state.mapping_edit_mode:
            c1, c2, _ = st.columns([1.1, 1.1, 3.8])
            with c1:
                with st.container(key="confirm_mapping"):
                    if st.button("Confirm structure", key="confirm_mapping_button", use_container_width=True):
                        if not st.session_state.get("user_col"):
                            st.error("Confirm a user/customer identifier before continuing.")
                        elif grain == "event" and (not st.session_state.get("event_col") or not st.session_state.get("timestamp_col")):
                            st.error("Event data needs a confirmed Event and Event time field.")
                        elif grain == "ambiguous":
                            st.warning("The row grain is ambiguous. Review mappings or continue only for data-quality inspection.")
                            st.session_state.workflow_step = "workspace"; st.session_state.workspace_page = "summary"; st.rerun()
                        else:
                            with st.spinner("Preparing the dataset once for fast analysis…"):
                                _finalize_dataset_for_workspace(df)
                            st.session_state.semantic_model = build_semantic_model(df)
                            st.session_state.workflow_step = "workspace"; st.session_state.workspace_page = "summary"; st.rerun()
            with c2:
                if st.button("Review mappings", key="edit_mapping_button", use_container_width=True):
                    st.session_state.mapping_edit_mode = True; st.rerun()
        else:
            all_columns = list(df.columns)
            optional_options = ["Not detected"] + all_columns
            with st.form("mapping_editor"):
                st.markdown("#### Core fields")
                new_user_col = st.selectbox("User identifier", options=all_columns, index=all_columns.index(st.session_state.user_col) if st.session_state.user_col in all_columns else 0)
                if grain == "event":
                    new_event_col = st.selectbox("Event", options=all_columns, index=all_columns.index(st.session_state.event_col) if st.session_state.event_col in all_columns else 0)
                else:
                    new_event_col = None
                time_options = ["Not detected"] + all_columns
                current_time = st.session_state.timestamp_col if st.session_state.timestamp_col in all_columns else "Not detected"
                selected_time = st.selectbox("Event / record time", options=time_options, index=time_options.index(current_time))
                new_timestamp_col = None if selected_time == "Not detected" else selected_time

                st.markdown("#### Optional analytical fields")
                edited_optional = {}
                roles = list(OPTIONAL_ROLE_LABELS.keys())
                for row_start in range(0, len(roles), 2):
                    row_cols = st.columns(2)
                    for offset, role in enumerate(roles[row_start:row_start + 2]):
                        current = st.session_state.optional_mappings.get(role)
                        current_value = current if current in all_columns else "Not detected"
                        with row_cols[offset]:
                            selected = st.selectbox(OPTIONAL_ROLE_LABELS[role], options=optional_options, index=optional_options.index(current_value), key=f"optional_map_{role}")
                            edited_optional[role] = None if selected == "Not detected" else selected

                edited_event_roles = {}
                if grain == "event":
                    st.markdown("#### Event roles")
                    event_values = ["Not detected"] + _event_values(df)
                    for row_start in range(0, len(EVENT_ROLE_LABELS), 2):
                        row_cols = st.columns(2)
                        for offset, role in enumerate(list(EVENT_ROLE_LABELS.keys())[row_start:row_start + 2]):
                            current = st.session_state.event_role_mappings.get(role)
                            current_value = current if current in event_values else "Not detected"
                            with row_cols[offset]:
                                selected = st.selectbox(EVENT_ROLE_LABELS[role], options=event_values, index=event_values.index(current_value), key=f"event_role_{role}")
                                edited_event_roles[role] = None if selected == "Not detected" else selected
                save_mapping = st.form_submit_button("Save mappings", type="primary", use_container_width=True)

            if save_mapping:
                core_values = [x for x in [new_user_col, new_event_col, new_timestamp_col] if x]
                optional_values = [v for v in edited_optional.values() if v]
                duplicates = sorted({x for x in core_values + optional_values if (core_values + optional_values).count(x) > 1})
                if duplicates:
                    st.error("A source column can map to only one analytical role. Please review: " + ", ".join(duplicates))
                else:
                    st.session_state.user_col = new_user_col
                    st.session_state.event_col = new_event_col
                    st.session_state.timestamp_col = new_timestamp_col
                    st.session_state.optional_mappings = edited_optional
                    st.session_state.custom_dimensions = detect_custom_dimensions(_inference_sample(df), edited_optional)
                    _invalidate_analysis_cache()
                    if grain == "event":
                        st.session_state.event_role_mappings = edited_event_roles
                        st.session_state.event_role_confidence = {r: ("Confirmed" if v else "Not detected") for r, v in edited_event_roles.items()}
                    st.session_state.data_quality = _initial_quality_report()
                    st.session_state.data_quality_finalized = False
                    st.session_state.timestamp_invalid_count = None
                    st.session_state.semantic_model = build_semantic_model(df)
                    st.session_state.mapping_edit_mode = False
                    st.rerun()

        st.markdown("")
        if st.button("← Choose another dataset", key="mapping_choose_other"):
            reset_dataset(); st.rerun()


# ================================================================
# WORKSPACE
# ================================================================

elif st.session_state.workflow_step == "workspace":
    df = st.session_state.dataset_df
    if not st.session_state.get("data_quality_finalized"):
        with st.spinner("Preparing the dataset once for fast analysis…"):
            _finalize_dataset_for_workspace(df)

    st.html(
        f"""
<div class="dataset-strip">
    <div class="dataset-dot"></div>
    <div>
        <div class="dataset-name">{st.session_state.dataset_name}</div>
        <div class="dataset-meta">{len(df):,} rows loaded</div>
    </div>
</div>
"""
    )


    workspace_navigation()

    if st.session_state.workspace_page == "summary":
        summary = get_dataset_summary()
        areas = _get_analysis_areas_cached()
        quality = st.session_state.get("data_quality") or build_data_quality_report(df)
        grain_info = st.session_state.get("grain_info") or {}
        ready_count = sum(area["status"] == "ready" for area in areas.values())
        area_cards = ""
        for area in areas.values():
            status = area["status"]
            label = {"ready": "Ready", "limited": "Limited", "missing": "Needs more data"}[status]
            symbol = {"ready": "✓", "limited": "◐", "missing": "○"}[status]
            reason = area["missing"] or "Available with the current dataset."
            area_cards += f'<div class="analysis-capability {status}"><div class="cap-status">{symbol}&nbsp; {label}</div><div class="cap-title">{area["label"]}</div><div class="cap-desc">{area["purpose"]}</div><div class="cap-reason">{reason}</div></div>'

        if summary["grain"] == "event":
            cards = f'<div class="summary-card"><div class="summary-label">Events</div><div class="summary-value">{summary["events"]:,}</div></div><div class="summary-card"><div class="summary-label">Users</div><div class="summary-value">{summary["users"]:,}</div></div><div class="summary-card"><div class="summary-label">Event types</div><div class="summary-value">{summary["event_types"]:,}</div></div><div class="summary-card"><div class="summary-label">Coverage</div><div class="summary-value">{summary["coverage"]}</div></div>'
        else:
            cards = f'<div class="summary-card"><div class="summary-label">Rows</div><div class="summary-value">{summary["rows"]:,}</div></div><div class="summary-card"><div class="summary-label">Users / customers</div><div class="summary-value">{summary["users"]:,}</div></div><div class="summary-card"><div class="summary-label">Detected grain</div><div class="summary-value">{summary["grain"].title()}</div></div><div class="summary-card"><div class="summary-label">Coverage</div><div class="summary-value">{summary["coverage"]}</div></div>'

        st.html(f"""<div class="lens-panel"><div class="lens-title">Dataset Summary</div><div class="lens-caption">LENS profiles the table before choosing metrics. Detected data type: <b>{summary['grain'].title()}</b>.</div><div class="summary-grid">{cards}</div><div class="analysis-readiness"><div class="readiness-header"><div><div class="readiness-title">Dashboard Readiness</div><div class="readiness-subtitle">Only analyses compatible with this dataset are unlocked.</div></div><div class="readiness-score">{ready_count} OF {len(areas)} READY</div></div><div class="analysis-grid">{area_cards}</div></div></div>""")
        if quality["issues"]:
            # Keep Streamlit's expander behavior, but force readable contrast on the dark workspace.
            st.html("""
<style>
[data-testid="stExpander"] {
  background: rgba(10, 19, 52, 0.34) !important;
  border: 1px solid rgba(255, 255, 255, 0.08) !important;
  border-radius: 16px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
  color: #F7F8FF !important;
  font-weight: 650 !important;
}
[data-testid="stExpander"] summary svg {
  color: #73F3EC !important;
  fill: #73F3EC !important;
}
.data-quality-item {
  padding: 11px 4px 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.07);
  font-size: 0.98rem;
  line-height: 1.55;
}
.data-quality-item:last-child { border-bottom: 0; }
.data-quality-title {
  color: #F7F8FF !important;
  font-weight: 750;
}
.data-quality-detail {
  color: #C9D2EB !important;
}
.data-quality-item.warning .data-quality-title { color: #FF79C6 !important; }
.data-quality-item.high .data-quality-title { color: #FF4FA3 !important; }
.data-quality-item.context .data-quality-title { color: #73F3EC !important; }
</style>
""")
            with st.expander(f"Data quality · {len(quality['issues'])} item(s)"):
                for issue in quality["issues"][:15]:
                    severity = str(issue.get("severity", "warning")).lower()
                    st.html(
                        f'<div class="data-quality-item {escape(severity)}">'
                        f'<span class="data-quality-title">{escape(str(issue["title"]))}</span>'
                        f'<span class="data-quality-detail"> — {escape(str(issue["detail"]))}</span>'
                        f'</div>'
                    )

    elif st.session_state.workspace_page == "analysis":
        areas = _get_analysis_areas_cached()
        if st.session_state.analysis_area not in areas:
            st.session_state.analysis_area = next(iter(areas.keys()))
        render_findings_overview(df)
        st.html('<div id="evidence-layer" class="evidence-scroll-anchor"></div><div class="analysis-explore-heading"><div class="analysis-explore-kicker">EVIDENCE LAYER</div><div class="analysis-explore-title">Explore analyses supported by this dataset</div></div>')

        active_area = st.session_state.analysis_area
        st.html(f"""<style>.st-key-analysis_area_{active_area} button {{color:#071D28 !important;background:linear-gradient(135deg,#55E8E0 0%,#3ED4D3 42%,#7A66F2 118%) !important;border-color:rgba(101,239,231,.60) !important;box-shadow:0 10px 28px rgba(70,225,218,.20),0 0 22px rgba(139,92,246,.10) !important;transform:translateY(-1px);}}</style>""")
        area_order = list(areas.keys())
        with st.container(key="analysis_area_nav"):
            cols = st.columns(len(area_order))
            for col, area_key in zip(cols, area_order):
                area = areas[area_key]
                button_label = area["label"] + (" · unavailable" if area["status"] == "missing" else " · limited" if area["status"] == "limited" else "")
                with col:
                    if st.button(button_label, key=f"analysis_area_{area_key}", use_container_width=True):
                        st.session_state.analysis_area = area_key; st.rerun()

        current = areas[st.session_state.analysis_area]
        status = current["status"]
        status_label = {"ready": "Ready", "limited": "Limited", "missing": "Needs more data"}[status]
        st.html(f'<div class="analysis-status-line"><div class="analysis-status {status}">● {status_label}</div><div class="analysis-status-hint">{current["purpose"]}</div></div>')
        if status == "missing":
            st.warning(current["missing"] or "This view needs additional data.")
        else:
            render_analysis_dashboard(st.session_state.analysis_area, df)

        if st.session_state.scroll_to_evidence:
            components.html("""<script>(function(){let tries=0;const timer=setInterval(function(){tries+=1;const doc=window.parent.document;const target=doc.getElementById('evidence-layer')||doc.querySelector('[class*="st-key-analysis_area_nav"]');if(target){target.scrollIntoView({behavior:'smooth',block:'start'});clearInterval(timer);}else if(tries>40){clearInterval(timer);}},100);})();</script>""", height=0, width=0)
            st.session_state.scroll_to_evidence = False

    elif st.session_state.workspace_page == "ask_lens":
        with st.container(key="ask_panel"):
            st.markdown("## Ask LENS")
            st.caption("Ask business questions about your product data in natural language.")

            question = st.text_area(
                "Ask a question",
                placeholder=(
                    "Why did activation drop after August?\n"
                    "Which segment has the weakest retention?\n"
                    "Where are users dropping in the funnel?"
                ),
                height=145,
            )

            if st.button("Ask LENS", key="ask_lens_button") and question.strip():
                st.info("Ask LENS will use the confirmed semantic model and deterministic metric functions. Natural-language execution is the remaining layer to connect.")

    floating_data_controls()


# ================================================================
# FOOTER
# ================================================================

st.html(
    """
<div class="lens-footer">
    <div>LENS · Product Analytics Intelligence Layer</div>
    <div>Designed &amp; built by <strong>SENA ESER</strong></div>
</div>
"""
)
