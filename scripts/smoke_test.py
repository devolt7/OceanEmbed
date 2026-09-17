"""Smoke test mirroring the deployed Streamlit Cloud app: boot both views without
exceptions and exercise prediction, replay, baselines and CSV export."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _check_boot(seeded: bool) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=600)
    if seeded:
        # seed a demo pill so the profile view runs (also skips the st.pills widget-state
        # quirk that the testing harness hits when a pill's value is None).
        at.session_state["demo_spot"] = "Bay of Bengal (cyclone zone)"
    at.run()
    assert not at.exception, f"AppTest raised: {[str(e.value)[:300] for e in at.exception]}"
    assert not at.error, f"AppTest errored: {[str(e.value)[:300] for e in at.error]}"
    if seeded:
        assert at.get("download_button"), "profile CSV download button missing"
        assert len(at.get("plotly_chart")) >= 3, "expected profile, replay + thermocline charts"
    print(f"{'profile' if seeded else 'map'} boot: OK")


def _check_inference() -> None:
    from src.utils.config import load_config
    from app.streamlit_app import baseline_eval, build_features, load_assets
    from app.analysis import mlp_predict, replay_monthly

    cfg = load_config()
    assets = load_assets(cfg)
    assert not assets["missing"], f"missing artifacts: {assets['missing']}"
    assert assets["mlp"] is not None and assets["grids"] and assets["collocated"] is not None

    X, used = build_features(assets["meta"], assets["grids"], 14.4, 71.6,
                             ref_time="2025-09-30")
    assert X is not None and len(used) == 6, f"bad feature build: {used}"
    pred = mlp_predict(assets["meta"], assets["scaler"], assets["mlp"], X.to_numpy())[0]
    assert 0 < float(pred[0]) < 40, "unphysical surface temperature"
    assert X.shape[1] == len(assets["meta"]["features"])

    rep = replay_monthly(cfg, assets["meta"], assets["scaler"], assets["mlp"], 14.4, 71.6)
    assert rep is not None and len(rep["months"]) >= 3, "replay broke"
    assert rep["pred"].shape[0] == len(rep["months"])

    rmse = baseline_eval(cfg, assets["meta"], assets["mlp"], assets["scaler"], None,
                         assets["collocated"])
    assert rmse is not None and len(rmse) == 3, "baseline RMSE table broke"
    print("inference / replay / baselines: OK")


if __name__ == "__main__":
    _check_boot(seeded=False)
    _check_boot(seeded=True)
    _check_inference()
    print("SMOKE OK")