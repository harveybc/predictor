
# This file contains the visualizer plugin, th input plugin can load all the data or starting from
 # the last id.

from flask import Blueprint
from flask import flash
from flask import g
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.exceptions import abort
from flask_login import login_required
from flask_login import current_user
from app.db import get_db
from flask import current_app
from flask import jsonify


def user_bp(plugin_folder):

    # construct the visualizer blueprint using the plugin folder as template folder
    bp = Blueprint("user_bp", __name__,  template_folder=plugin_folder)

    @bp.route("/users")
    @login_required
    def user_index():
        """Show the users index."""
        user_list = current_app.config['FE'].ep_input.get_users()
        return render_template("/plugin_templates/user/index.html", user_list = user_list)

    @bp.route("/user/<username>")
    @login_required
    def user_detail(username):
        """Show the user detail view, if it is the current user, shows a change password button."""
        user_list = current_app.config['FE'].ep_input.get_user_by_username(username)
        return render_template("/plugin_templates/user/detail.html", user_list =  user_list, username = username)

 
    return bp
