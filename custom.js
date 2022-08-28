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

// nbextensions Snippets Menu 커스텀
require(["nbextensions/snippets_menu/main"], function (snippets_menu) {
  console.log("Loading `snippets_menu` customizations from `custom.js`");
  var horizontal_line = "---";
  var tips = {
    name: "Tips",
    "sub-menu": [
      {
        name: "이미지 삽입",
        snippet: ["![](img/file_name)  # 마크다운에서 실행"],
      },
    ],
  };
  var pandas_custom = {
    name: "pandas_custom",
    "sub-menu": [
      {
        name: "First check",
        snippet: [
          "df.shape",
          "df.head()",
          "df.tail()",
          "df.dtypes",
          "df.info()",
          "df.describe()",
        ],
      },
      {
        name: "Basic stats",
        "sub-menu": [
          {
            name: "quantile",
            snippet: ["bp_sr.quantile()", "bp_df.quantile()"],
          },
        ],
      },
    ],
  };
  var pydata_book = {
    name: "pydata_book",
    "sub-menu": [
      {
        name: "Setup",
        snippet: [
          "import numpy as np",
          "import pandas as pd",
          "PREVIOUS_MAX_ROWS = pd.options.display.max_rows",
          "pd.options.display.max_rows = 20",
          "np.random.seed(12345)",
          "import matplotlib.pyplot as plt",
          "plt.rc('figure', figsize=(10, 6))",
          "np.set_printoptions(precision=4, suppress=True)",
        ],
      },
    ],
  };
  var scikit_learn = {
    name: "scikit_learn",
    "sub-menu": [
      { // pipeline
        name: "pipeline",
        "sub-menu": [
          {
            name: "Pipeline",
            snippet: [
              "from sklearn.pipeline import Pipeline",
              "",
              "polynomial_regression = Pipeline([",
              "    ('poly_features', PolynomialFeatures(degree = 10, include_bias = False)),",
              "    ('lin_reg', LinearRegression()),",
              "])  # 2장? 3장? 참조해서 수정 필요",
            ],
          },
        ],
      },
      { // early stopping
        name: "early stopping",
        "sub-menu": [
          {
            name: "implementation",
            snippet: [
              "from copy import deepcopy",
              "",
              "poly_scaler = Pipeline([",
              "    ('poly_features', PolynomialFeatures(degree=90, include_bias=False)),",
              "    ('std_scaler', StandardScaler())",
              "])",
              "",
              "X_train_poly_scaled = poly_scaler.fit_transform(X_train)",
              "X_val_poly_scaled = poly_scaler.transform(X_val)",
              "",
              "sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,  # warm_start=True: fit() 메서드가 호출될 때 처음부터 다시 시작하지 않고 이전 모델 파라미터에서 훈련을 이어간다.",
              "                       penalty=None, learning_rate='constant', eta0=0.0005, random_state=42)",
              "",
              "minimum_val_error = float('inf')",
              "best_epoch = None",
              "best_model = None",
              "for epoch in range(1000):",
              "    sgd_reg.fit(X_train_poly_scaled, y_train)  # 중지된 곳에서 다시 시작합니다",
              "    y_val_predict = sgd_reg.predict(X_val_poly_scaled)",
              "    val_error = mean_squared_error(y_val, y_val_predict)",
              "    if val_error < minimum_val_error:",
              "        minimum_val_error = val_error",
              "        best_epoch = epoch",
              "        best_model = deepcopy(sgd_reg)",
            ],
          },
          {
            name: "graph",
            snippet: [
              "sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,",
              "         penalty=None, learning_rate='constant', eta0=0.0005, random_state=42)",
              "",
              "n_epochs = 500",
              "train_errors, val_errors = [], []",
              "for epoch in range(n_epochs):",
              "    sgd_reg.fit(X_train_poly_scaled, y_train)",
              "    y_train_predict = sgd_reg.predict(X_train_poly_scaled)",
              "    y_val_predict = sgd_reg.predict(X_val_poly_scaled)",
              "    train_errors.append(mean_squared_error(y_train, y_train_predict))",
              "    val_errors.append(mean_squared_error(y_val, y_val_predict))",
              "",
              "best_epoch = np.argmin(val_errors)",
              "best_val_rmse = np.sqrt(val_errors[best_epoch])",
              "",
              "plt.annotate('Best model',",
              "            xy=(best_epoch, best_val_rmse),",
              "            xytext=(best_epoch, best_val_rmse + 1),",
              "            ha='center',",
              "            arrowprops=dict(facecolor='black', shrink=0.05),",
              "            fontsize=16,",
              "            )",
              "",
              "best_val_rmse -= 0.03  # just to make the graph look better",
              "plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], 'k:', linewidth=2)",
              "plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='Validation set')",
              "plt.plot(np.sqrt(train_errors), 'r--', linewidth=2, label='Training set')",
              "plt.legend(loc='upper right', fontsize=14)",
              "plt.xlabel('Epoch', fontsize=14)",
              "plt.ylabel('RMSE', fontsize=14)",
              "plt.show()",
            ],
          },
        ],
      },
      { // data set
        name: "data set",
        "sub-menu": [
          {
            name: "moons",
            snippet: [
              "from sklearn.model_selection import train_test_split",
              "from sklearn.datasets import make_moons",
              "",
              "X, y = make_moons(n_samples = 500, noise = 0.30, random_state = 42)",
              "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)",
            ],
          },
          {
            name: "잡음이 섞인 2차 곡선",
            snippet: [
              "np.random.seed(42)",
              "X = np.random.rand(100, 1) - 0.5",
              "y = 3 * X[: , 0] ** 2 + 0.05 * np.random.randn(100)",
            ],
          },
        ],
      },
      { // linear_model
        name: "linear_model",
        "sub-menu": [
          {
            name: "LinearRegression",
            snippet: [
              "from sklearn.linear_model import LinearRegression",
              "",
              "lin_reg = LinearRegression()",
              "lin_reg.fit(X_train, y_train)",
              "# lin_reg.intercept_, lin_reg.coef_  # suffix underscore: 학습된 모델 파라미터",
              "# lin_reg.predict(X_train)",
            ],
          },
          {
            name: "LogisticRegression",
            snippet: [
              "from sklearn.linear_model import LogisticRegression",
              "",
              "log_reg = LogisticRegression(random_state=42)",
              "log_reg.fit(X_train, y_train)",
              "log_reg.predict(X_train)",
              "log_reg.predict_proba(X_train)",
            ],
          },
          {
            name: "SGDRegressor",
            snippet: [
              "from sklearn.linear_model import SGDRegressor",
              "",
              "sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)",
              "# 최대 100번의 에포크(max_iter = 1000) 동안 실행되거나 한 에포크에서 0.001보다 적게 손실이 줄어들 때까지 실행(tol = 1e-3)",
              "# penalty = None: 규제는 사용하지 않음",
              "# 학습률 0.1(eta0 = 0.1)로 기본 학습 스케줄 사용",
              "# penalty='l2' 인자를 추가하면 확률적 경사 하강법을 사용한 릿지 회귀를 수행한다",
              "# penalty='l1' 인자를 추가하면 확률적 경사 하강법을 사용한 라쏘 회귀를 수행한다",
              "# sgd_reg.intercept_, sgd_reg.coef_",
              "sgd_reg.fit(X_train, y_train)",
            ],
          },
          {
            name: "SGDClassifier",
            snippet: [
              "from sklearn.linear_model import SGDClassifier",
              "",
              "sgd_clf = SGDClassifier()",
            ],
          },
          {
            name: "Ridge",
            snippet: [
              "from sklearn.linear_model import Ridge",
              "",
              "# 정규방정식을 사용한 릿지 회귀",
              "ridge_reg = Ridge(alpha=1, solver='cholesky', random_state=42)  # cholesky: 숄레스키?",
              "",
              "# sag를 사용한 릿지 회귀",
              "# ridge_reg = Ridge(alpha=1, solver='sag', random_state=42)  # 'sag': stochastic average gradient descent (확률적 평균 경사 하강법) - sgd의 변종",
              "",
              "ridge_reg.fix(X, y)",
              "ridge_reg.predict(X)",
            ],
          },
          {
            name: "Lasso",
            snippet: [
              "from sklearn.linear_model import Lasso",
              "",
              "lasso_reg = lasso(alpha=0.1)",
              "lasso_reg.fix(X, y)",
              "lasso_reg.predict(X)",
            ],
          },
          {
            name: "ElasticNet",
            snippet: [
              "from sklearn.linear_model import ElasticNet",
              "",
              "elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)",
              "elastic_net.fix(X, y)",
              "elastic_net.predict(X)",
            ],
          },
        ],
      },
      { // model_selection
        name: "model_selection",
        "sub-menu": [
          {
            name: "train_test_split",
            snippet: [
              "from sklearn.model_selection import train_test_split",
              "",
              "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)",
            ],
          },
          {
            name: "cross_val_predict",
            snippet: [
              "from sklearn.model_selection import cross_val_predict",
              "",
              "y_train_pred = cross_val_predict(model, X_train, y_train, cv=3)  # 예측값을 반환",
              "y_train_scores = cross_val_predict(model, X_train, y_train, cv=3,",
              "                                method = 'decision_function')  # 점수를 반환",
              "                                                               # 여기서 점수는 분류기의 성능이 아니라, 분류에 사용할 점수",
            ],
          },
          {
            name: "cross_val_score",
            snippet: [
              "from sklearn.model_selection import cross_val_score",
              "",
              "cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')  # model의 성능 점수를 반환",
            ],
          },
        ],
      },
      { // preprocessing
        name: "preprocessing",
        "sub-menu": [
          {
            name: "StandardScaler",
            snippet: [
              "from sklearn.preprocessing import StandardScaler",
              "",
              "scaler = StandardScaler()",
              "X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))",
            ],
          },
          {
            name: "PolynomialFeatures",
            snippet: [
              "from sklearn.preprocessing import PolynomialFeatures",
              "",
              "poly_features = PolynomialFeatures(degree=2, include_bias=True)",
              "X_poly = poly_features.fit_transform(X)",
              "",
              "# lin_reg = LinearRegression()",
              "# lin_reg.fit(X_poly, y)",
              "# lin_reg.intercept_, lin_reg.coef_",
            ],
          },
        ],
      },
      { // svm
        name: "svm",
        "sub-menu": [
          {
            name: "LinearSVC",
            snippet: [
              "from sklearn.svm import LinearSVC",
              "",
              "svm_clf = Pipeline([",
              "        ('scaler', StandardScaler()),",
              "        ('linear_svc', LinearSVC(C=1, loss='hinge'))",
              "    ])",
              "svm_clf.fit(X, y)",
            ],
          },
          {
            name: "SVC",
            snippet: [
              "from sklearn.svm import SVC",
              "",
              "svm_clf = SVC(gamma='auto', random_state=42)",
              "svm_clf.fit(X_train, y_train)",
              "svm_clf.predict(X_train)",
            ],
          },
          {
            name: "SVC  # with polynomial kernel trick",
            snippet: [
              "# homl 212/953, 5.2.1 다항식 커널",
              "from sklearn.svm import SVC",
              "",
              "poly_kernel_svm_clf = Pipeline([",
              "        ('scaler', StandardScaler()),",
              "        ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))",
              "    ])",
              "poly_kernel_svm_clf.fit(X, y)",
            ],
          },
          {
            name: "SVC  # with Gaussian rbf kernel trick",
            snippet: [
              "from sklearn.svm import SVC",
              "",
              "rbf_kernel_svm_clf = Pipeline([",
              "        ('scaler', StandardScaler()),",
              "        ('svm_clf', SVC(kernel='rbf', gamma=5, C=0.001))",
              "    ])",
              "rbf_kernel_svm_clf.fit(X, y)",
            ],
          },
          {
            name: "LinearSVR",
            snippet: [
              "from sklearn.svm import LinearSVR",
              "",
              "svm_reg = LinearSVR(epsilon = 1.5, random_state = 42)",
              "svm_reg.fit(X, y)",
            ],
          },
          {
            name: "LinearSVR  # with polynomial kernel trick",
            snippet: [
              "from sklearn.svm import SVR",
              "",
              "svm_poly_reg = SVR(kernel = 'poly', degree = 2, C = 100, epsilon = 0.1, gamma = 'scale')",
              "svm_poly_reg.fit(X, y)",
            ],
          },
        ],
      },
      { // Decision Tree
        name: "Decision Tree",
        "sub-menu": [
          {
            name: "DecisionTreeClassifier",
            snippet: [
              "from sklearn.datasets import load_iris",
              "from sklearn.tree import DecisionTreeClassifier",
              "",
              "iris = load_iris()",
              "X = iris.data[:, 2: ]  # 꽃잎 길이와 너비",
              "y = iris.target",
              "",
              "tree_clf = DecisionTreeClassifier(max_depth = 2, random_state = 42)",
              "tree_clf.fit(X, y)",
            ],
          },
          {
            name: "DecisionTreeRegressor",
            snippet: [
              "from sklearn.tree import DecisionTreeRegressor",
              "",
              "tree_reg = DecisionTreeRegressor(max_depth = 2, random_state = 42)",
              "tree_reg.fit(X, y)",
            ],
          },
        ],
      },
      { // neighbors
        name: "neighbors",
        "sub-menu": [
          {
            name: "KNeighborsClassifier",
            snippet: [
              "from sklearn.neighbors import KNeighborsClassifier",
              "",
              "knn_clf = KNeighborsClassifier()",
              "knn_clf.fit(X_train, y_train)",
              "knn_clf.predict(X_train)",
            ],
          },
        ],
      },
      { // multiclass
        name: "multiclass",
        "sub-menu": [
          {
            name: "OneVsRestClassifier",
            snippet: [
              "# OvO대신 OvR 전략을 사용하는 분류기",
              "from sklearn.multiclass import OneVsRestClassifier",
              "",
              "ovr_clf = OneVsRestClassifier(SVC(gamma = 'auto', random_state = 42))  # 다중 분류에서 SVC가 OvO가 아닌 OvR 전략을 사용하도록 강제",
              "ovr_clf.fit(X_train, y_train)",
              "ovr_clf.predict(X_train)",
            ],
          },
        ],
      },
      { // ensemble
        name: "ensemble",
        "sub-menu": [
          {
            name: "RandomForestClassifier",
            snippet: [
              "from sklearn.ensemble import RandomForestClassifier",
              "",
              "forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)",
              "y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,",
              "                                    method='predict_proba')",
            ],
          },
          {
            name: "RandomForestClassifier",
            snippet: [
              "# homl 255/953",
              "from sklearn.ensemble import RandomForestClassifier",
              "",
              "rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, random_state = 42)",
              "rnd_clf.fit(X_train, y_train)",
              "rnd_clf.feature_importances_",
              "",
              "y_pred_rf = rnd_clf.predict(X_test)",
            ],
          },
          {
            name: "AdaBoostClassifier",
            snippet: [
              "# homl 263/953",
              "from sklearn.ensemble import AdaBoostClassifier",
              "",
              "ada_clf = AdaBoostClassifier(",
              "    DecisionTreeClassifier(max_depth = 1), n_estimators = 200,  # n_estimators: 추정기 수",
              "    algorithm = 'SAMME.R', learning_rate = 0.5, random_state = 42)",
              "ada_clf.fit(X_train, y_train)",
            ],
          },
          {
            name: "GradientBoostingRegressor",
            snippet: [
              "# homl 266/953",
              "from sklearn.ensemble import GradientBoostingRegressor",
              "",
              "gbrt = GradientBoostingRegressor(max_depth = 2, n_estimators = 3, learning_rate = 1.0, random_state = 42)",
              "gbrt.fit(X, y)",
            ],
          },
        ],
      },
      { // metircs
        name: "metrics",
        "sub-menu": [
          {
            name: "accuracy_score",
            snippet: [
              "from sklearn.metrics import accuracy_score",
              "",
              "accuracy_score(y_test, y_pred)",
            ],
          },
          {
            name: "mean_squared_error",
            snippet: [
              "from sklearn.metircs import mean_squared_error",
              "",
              "mean_squared_error(y_train, y_train_pred)",
            ],
          },
          {
            name: "confusion_matrix",
            snippet: [
              "from sklearn.metircs import confusion_matrix",
              "",
              "confusion_matrix(y_train, y_train_pred)",
            ],
          },
          {
            name: "precision_score",
            snippet: [
              "from sklearn.metrics import precision_score",
              "",
              "precision_score(y_bp, y_bp_pred)",
            ],
          },
          {
            name: "recall_score",
            snippet: [
              "from sklearn.metrics import recall_score",
              "",
              "recall_score(y_bp, y_bp_pred)",
            ],
          },
          {
            name: "f1_score",
            snippet: [
              "from sklearn.metrics import f1_score",
              "",
              "f1_score(y_bp, y_bp_pred)",
            ],
          },
          {
            name: "precision_recall_curve",
            snippet: [
              "from sklearn.metrics import precision_recall_curve",
              "",
              "precision, recalls, thresholds = precision_recall_curve(y_bp, y_bp_scores)",
            ],
          },
          {
            name: "roc_curve",
            snippet: [
              "from sklearn.metrics import roc_curve",
              "",
              "fpr, tpr, thresholds = roc_curve(y_bp, y_bp_scores)  # y_bp_scores는 classifier의 decision_function()으로 구함",
              "                                                     # decision_function()이 없는 classifier의 경우,",
              "                                                     # predict_proba()로 확률을 구하고 양성 클래스일 확률을 점수 대신 사용",
              "",
              "def plot_roc_curve(fpr, tpr, label=None):",
              "    plt.plot(fpr, tpr, linewidth=2, label=label)",
              "    plt.plot([0, 1], [0, 1], 'k--') # 대각 점선",
              "    plt.axis([0, 1, 0, 1])",
              "    plt.xlabel('False Positive Rate (Fall-Out)', fontsize = 16)",
              "    plt.ylabel('True Positive Rate (Recall)', fontsize = 16)",
              "    plt.grid(True)",
              "",
              "plt.figure(figsize=(8, 6))",
              "plot_roc_curve(fpr, tpr)",
              "fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]",
              "plt.plot([fpr_90, fpr_90], [0., recall_90_precision], 'r:')",
              "plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], 'r:')",
              "plt.plot([fpr_90], [recall_90_precision], 'ro')",
              "plt.show()",
            ],
          },
          {
            name: "roc_auc_score",
            snippet: [
              "from sklearn.metrics import roc_auc_score",
              "",
              "roc_auc_score(y_bp, y_bp_scores)  # y_bp_scores는 classifier의 decision_function()이나 predict_proba()로 계산",
            ],
          },
        ],
      },
      { // markdown
        name: "markdown",
        "sub-menu": [
          {
            name: "md: confusion_matrix",
            snippet: [
              "|실제|negative로 분류|positive로 분류|",
              "|---|---|---|",
              "|negative class| true negative(진짜 음성) | false positive(거짓 양성) |",
              "|positive class| false negative(거짓 음성) | true positive(진짜 양성) |",
            ],
          },
        ],
      },
      { // decomposition  # 차원 축소
        name: "decomposition  # 차원 축소",
        "sub-menu": [
          {
            name: "PCA",
            snippet: [
              "from sklearn.decomposition import PCA",
              "",
              "pca = PCA(n_components = 2)",
              "X2D = pca.fit_transform(X)",
            ],
          },
        ],
      },
      { // cluster
        name: "cluster",
        "sub-menu": [
          {
            name: "KMeans",
            snippet: [
              "from sklearn.cluster import KMeans",
              "k = 5",
              "kmeans = KMeans(n_clusters=k, random_state=42)",
              "y_pred = kmeans.fit_predict(X)",
              "",
              "kmeans.labels_",
              "kmeans.cluster_centers_",
            ],
          },
        ],
      },
    ],
  };
  var my_favorites = {
    name: "My $\\nu$ favorites",
    "sub-menu": [
      {
        name: "Multi-line snippet",
        snippet: [
          "new_command(3.14)",
          "",
          'other_new_code_on_new_line("with a string!")',
          "stringy('escape single quotes once')",
          "stringy2('or use single quotes inside of double quotes')",
          'backslashy("This \\ appears as just one backslash in the output")',
          'backslashy2("Here are \\\\ two backslashes")',
        ],
      },
      {
        name: "code test",
        snippet: [
          "import unittest",
          "",
          "class bp_TestCase(unittest.TestCase):",
          "    def test_bp(self):",
          "        self.assertEqual()",
          "        self.assertNotEqual()",
          "        self.assertTrue()",
          "        self.assertFalse()",
          "        self.assertIn()",
          "        self.assertNotIn()",
          "",
          "if __name__ == '__main__':",
          "    unittest.main()",
        ],
      },
    ],
  };
  snippets_menu.default_menus[0]["sub-menu"].splice(1, 1); // Remove Scipy
  snippets_menu.default_menus[0]["sub-menu"].splice(2, 1); // Remove Sympy
  snippets_menu.default_menus[0]["sub-menu"].splice(3, 3); // Remove Astropy, h5py, and numba (3, 3): 3 위치부터 3개를 지움
  snippets_menu.options["menus"].push(snippets_menu.default_menus[0]); // Start with the remaining "Snippets" menu
  snippets_menu.options["menus"][0]["sub-menu"].push(horizontal_line);
  snippets_menu.options["menus"][0]["sub-menu"].push(tips);
  snippets_menu.options["menus"][0]["sub-menu"].push(pandas_custom);
  snippets_menu.options["menus"][0]["sub-menu"].push(pydata_book);
  snippets_menu.options["menus"][0]["sub-menu"].push(scikit_learn);
  snippets_menu.options["menus"][0]["sub-menu"].push(my_favorites);
  console.log("Loaded `snippets_menu` customizations from `custom.js`");
});