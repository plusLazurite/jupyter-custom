// leave at least 2 line with only a star on it below, or doc generation fails
/**
 *
 *
 * Placeholder for custom user javascript
 * mainly to be overridden in profile/static/custom/custom.js
 * This will always be an empty file in IPython
 *
 * User could add any javascript in the `profile/static/custom/custom.js` file.
 * It will be executed by the ipython notebook at load time.
 *
 * Same thing with `profile/static/custom/custom.css` to inject custom css into the notebook.
 *
 *
 * The object available at load time depend on the version of IPython in use.
 * there is no guaranties of API stability.
 *
 * The example below explain the principle, and might not be valid.
 *
 * Instances are created after the loading of this file and might need to be accessed using events:
 *     define([
 *        'base/js/namespace',
 *        'base/js/events'
 *     ], function(IPython, events) {
 *         events.on("app_initialized.NotebookApp", function () {
 *             IPython.keyboard_manager....
 *         });
 *     });
 *
 * __Example 1:__
 *
 * Create a custom button in toolbar that execute `%qtconsole` in kernel
 * and hence open a qtconsole attached to the same kernel as the current notebook
 *
 *    define([
 *        'base/js/namespace',
 *        'base/js/events'
 *    ], function(IPython, events) {
 *        events.on('app_initialized.NotebookApp', function(){
 *            IPython.toolbar.add_buttons_group([
 *                {
 *                    'label'   : 'run qtconsole',
 *                    'icon'    : 'icon-terminal', // select your icon from http://fortawesome.github.io/Font-Awesome/icons
 *                    'callback': function () {
 *                        IPython.notebook.kernel.execute('%qtconsole')
 *                    }
 *                }
 *                // add more button here if needed.
 *                ]);
 *        });
 *    });
 *
 * __Example 2:__
 *
 * At the completion of the dashboard loading, load an unofficial javascript extension
 * that is installed in profile/static/custom/
 *
 *    define([
 *        'base/js/events'
 *    ], function(events) {
 *        events.on('app_initialized.DashboardApp', function(){
 *            require(['custom/unofficial_extension.js'])
 *        });
 *    });
 *
 * __Example 3:__
 *
 *  Use `jQuery.getScript(url [, success(script, textStatus, jqXHR)] );`
 *  to load custom script into the notebook.
 *
 *    // to load the metadata ui extension example.
 *    $.getScript('/static/notebook/js/celltoolbarpresets/example.js');
 *    // or
 *    // to load the metadata ui extension to control slideshow mode / reveal js for nbconvert
 *    $.getScript('/static/notebook/js/celltoolbarpresets/slideshow.js');
 *
 *
 * @module IPython
 * @namespace IPython
 * @class customjs
 * @static
 */

// stackoverflow: Disable Ctrl+Enter sublime keymap in jupyter notebook
 require(["codemirror/keymap/sublime", "notebook/js/cell", "base/js/namespace"],
 function(sublime_keymap, cell, IPython) {
     cell.Cell.options_default.cm_config.keyMap = 'sublime';
     cell.Cell.options_default.cm_config.extraKeys["Ctrl-Enter"] = function(cm) {}
     var cells = IPython.notebook.get_cells();
     for(var cl=0; cl< cells.length ; cl++){
         cells[cl].code_mirror.setOption('keyMap', 'sublime');
         cells[cl].code_mirror.setOption("extraKeys", {
             "Ctrl-Enter": function(cm) {}
         });
     }
 } 
);

// Register a global action (navigation(menu) bar) 토글 기능 추가
var action_name = Jupyter.actions.register({
    help: 'hide/show the menubar',
    handler : function(env) {
        $('#menubar').toggle();
        events.trigger('resize-header.Page');
    }
}, 'toggle-menubar', 'jupyter-notebook');
// Add a menu item to the View menu
$('#view_menu').prepend('<li id="toggle_menu" title="Show/Hide the menu bar"><a href="#">Toggle Menu</a></li>').click(function() {
    Jupyter.actions.call(action_name);
});
// Add a shortcut: CMD+M (or CTRL+M on Windows) to toggle menu bar
Jupyter.keyboard_manager.command_shortcuts.add_shortcut('N', action_name);