
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


def dashboard_bp(plugin_folder):

    # construct the visualizer blueprint using the plugin folder as template folder
    bp = Blueprint("dashboard_bp", __name__,  template_folder=plugin_folder)
    
    @bp.route("/")
    @login_required
    def index():
        # read the data to be visualized using the using the Feature extractor instance, preinitialized in __init__.py with input and output plugins entry points.
        # TODO: replace 0 in vis_data by process_id, obtained as the first process_id belonging to the current user.    
        # vis_data = current_app.config['FE'].ep_input.load_data(current_app.config['P_CONFIG'], 0)
        box= []
        print("user_id = ", current_user.id)
        box.append(current_app.config['FE'].ep_input.get_max(current_user.id, "training_progress", "mse"))
        box.append(current_app.config['FE'].ep_input.get_max(current_user.id, "validation_stats", "mse"))
        box.append(current_app.config['FE'].ep_input.get_count("user"))
        box.append(current_app.config['FE'].ep_input.get_count("process"))
        #TODO: Usar campo y tabla configurable desde JSON para graficar
        v_original = current_app.config['FE'].ep_input.get_column_by_pid("validation_plots", "original", box[0]['id'] )
        v_predicted = current_app.config['FE'].ep_input.get_column_by_pid("validation_plots", "predicted", box[0]['id'] )
        p,t,v = current_app.config['FE'].ep_input.processes_by_uid(current_user.id)
        #tr_data = current_app.config['FE'].ep_input.training_data("trainingprogress", "mse")
        status = []
        for i in range(0,len(p)):
            print ("v[i]['mse'] = ", v[i]['mse'])
            print ("t[i]['mse'] = ", t[i]['mse'])
            if v[i]['mse'] == None and t[i]['mse'] == None:
                status.append("Not Started")
                v[i]['MAX(mse)'] = 0.0
            elif v[i]['mse'] != None and t[i]['mse'] != None:
                status.append("Validation")           
            elif v[i]['mse'] == None and t[i]['mse'] != None: 
                v[i] = t[i]
                status.append("Training")
            print("status[",i,"] = ", status[i])
        return render_template("/plugin_templates/dashboard/index.html", p_config = current_app.config['P_CONFIG'], box = box, v_original = v_original, v_predicted = v_predicted, p=p, v=v, status=status)

    @bp.route("/<int:pid>/trainingpoints")
    def get_points(pid):
        """Get the points to plot from the training_progress table and return them as JSON."""
        xy_points = get_xy_training(pid)
        return jsonify(xy_points)

    def get_xy_training(pid):
        """ Returns the points to plot from the training_progress table. """
        results = current_app.config['FE'].ep_input.get_column_by_pid("training_progress", "mse", pid )
        return results

    


    @bp.route("/processes")
    @login_required
    def process_index():
        """Show the processes index."""
        process_list = current_app.config['FE'].ep_input.get_processes(current_user.id)
        return render_template("/plugin_templates/process/index.html", process_list = process_list)

    @bp.route("/process/<pid>")
    @login_required
    def process_detail(pid):
        """Show the process detail view, if it is the current user, shows a change password button."""
        process_list = current_app.config['FE'].ep_input.get_process_by_pid(pid)
        return render_template("/plugin_templates/process/detail.html", process_list = process_list, pid = pid)




    def get_post(id, check_author=True):
        """Get a post and its author by id.

        Checks that the id exists and optionally that the current user is
        the author.

        :param id: id of post to get
        :param check_author: require the current user to be the author
        :return: the post with author information
        :raise 404: if a post with the given id doesn't exist
        :raise 403: if the current user isn't the author
        """
        results = (
            get_db()
            .execute(
                "SELECT p.id, title, body, created, author_id, username"
                " FROM post p JOIN user u ON p.author_id = u.id"
                " WHERE p.id = ?",
                (id,),
            )
            .fetchone()
        )
        # verify if the query returned no results
        if results is None:
            abort(404, "Post id {id} doesn't exist.")
        return results

    return bp