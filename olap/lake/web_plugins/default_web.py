"""OLAP GUI + HTTP API for data-gov: SELECT for query, append-only gov_*
through POST /api/v1/metrics. Never runs reset_olap."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

from app.operator_config import save_pending
from app.lake_auth import check_bearer, load_token


def _fmt_bytes(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


class Plugin:
    plugin_params = {"web_host": "127.0.0.1", "web_port": 5057, "secret_key": "x"}

    def __init__(self):
        self.params = dict(self.plugin_params)
        self._context = None

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def create_app(self, context):
        self._context = context
        here = Path(__file__).resolve().parent
        app = Flask(
            __name__,
            template_folder=str(here / "templates"),
            static_folder=str(here / "static"),
            static_url_path="/static",
        )
        app.secret_key = self.params.get("secret_key") or "x"
        app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
        q = lambda: context["plugins"]["query"]
        cfg = lambda: context["config"]

        @app.context_processor
        def inject():
            return {"fmt_bytes": _fmt_bytes}

        @app.get("/healthz")
        def healthz():
            return "ok\n", 200, {"Content-Type": "text/plain"}

        @app.get("/")
        def home():
            try:
                resources = q().discover()
                err = None
            except Exception as exc:
                resources = []
                err = str(exc)
            return render_template(
                "home.html",
                meta=q().describe() if not err else cfg(),
                storage=q().storage(),
                resources=resources,
                error=err,
                holdout=cfg().get("holdout_start") or "",
                config=cfg(), result=None,
            )

        @app.post("/config")
        def save():
            try:
                dest = save_pending(cfg(), request.form)
            except (ValueError, TypeError, OSError) as exc:
                flash(f"Invalid configuration: {exc}", "danger")
            else:
                flash(f"Pending restart: {dest}. Active configuration unchanged.", "success")
            return redirect(url_for("home"))

        @app.post("/ops/query")
        def ops_query():
            sql = request.form.get("sql") or ""
            try:
                payload = q().query(sql)
            except (ValueError, PermissionError) as exc:
                flash(str(exc), "danger")
                return redirect(url_for("home"))
            return render_template(
                "home.html", meta=q().describe(), storage=q().storage(),
                resources=q().discover(), error=None, config=cfg(),
                holdout=cfg().get("holdout_start") or "", result=payload,
            )

        @app.get("/resources/<resource_id>")
        def resource_detail(resource_id):
            try:
                resource = q().resource_schema(resource_id)
            except ValueError as exc:
                return str(exc), 404
            return render_template("resource.html", resource=resource)

        def _api_ok():
            expected = cfg().get("lake_service_token") or load_token()
            if check_bearer(request.headers.get("Authorization"), expected):
                return None
            return jsonify({"error": "unauthenticated"}), 401

        @app.get("/api/v1/describe")
        def api_describe():
            denied = _api_ok()
            if denied:
                return denied
            return jsonify(q().describe())

        @app.get("/api/v1/storage")
        def api_storage():
            denied = _api_ok()
            if denied:
                return denied
            return jsonify(q().storage())

        @app.get("/api/v1/discover")
        def api_discover():
            denied = _api_ok()
            if denied:
                return denied
            try:
                return jsonify({"resources": q().discover()})
            except Exception as exc:
                return jsonify({"error": str(exc), "resources": []}), 503

        @app.get("/api/v1/query")
        def api_query():
            denied = _api_ok()
            if denied:
                return denied
            sql = request.args.get("sql") or ""
            try:
                return jsonify(q().query(sql))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except PermissionError:
                return jsonify({"error": "holdout"}), 403

        @app.post("/api/v1/metrics")
        def api_metrics():
            denied = _api_ok()
            if denied:
                return denied
            report = request.get_json(silent=True)
            if not isinstance(report, dict):
                return jsonify({"error": "report must be a JSON object"}), 400
            try:
                result = q().write_metrics(report)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except Exception as exc:
                return jsonify({"error": f"database error: {exc}"}), 503
            result["report_sha256"] = report.get("report_sha256")
            return jsonify(result), (201 if result.get("stored") else 200)

        @app.get("/api/v1/schema")
        def api_schema():
            denied = _api_ok()
            if denied:
                return denied
            try:
                return jsonify(q().resource_schema(request.args.get("resource", "")))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 404

        @app.route("/api/v2/terminals", methods=["GET", "POST"])
        def api_terminals():
            denied = _api_ok()
            if denied:
                return denied
            if request.method == "GET":
                campaign = request.args.get("campaign_sha256") or ""
                try:
                    return jsonify({"terminals": q().terminal_digests(campaign)})
                except ValueError as exc:
                    return jsonify({"error": str(exc)}), 400
                except Exception as exc:
                    return jsonify({"error": f"database error: {exc}"}), 503
            terminal = request.get_json(silent=True)
            if not isinstance(terminal, dict):
                return jsonify({"error": "terminal must be a JSON object"}), 400
            try:
                result = q().write_terminal(terminal)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except Exception as exc:
                return jsonify({"error": f"database error: {exc}"}), 503
            return jsonify(result), (201 if result.get("stored") else 200)

        return app

    def serve(self, context):
        app = self.create_app(context)
        host = self.params.get("web_host") or "127.0.0.1"
        port = int(self.params.get("web_port") or 5057)
        print(f"OLAP warehouse UI → http://{host}:{port}")
        print("SELECT for query; append-only gov_* through write_metrics; never reset_olap")
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
        return 0
