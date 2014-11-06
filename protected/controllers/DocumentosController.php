<?php

class DocumentosController extends Controller {

    public $layout = '//layouts/responsiveLayout';
    private $_model;

    public function filters() {
        return array(
            'accessControl',
        );
    }

    public function accessRules() {
        return array(
            array('allow',
                'actions' => array('index', 'view'),
                'users' => array('*'),
            ),
            array('allow',
                'actions' => array('create', 'update', 'CreateSubir','CreateFisico','CreateOnline','CreateSubirEquipo','CreateOnlineEquipo','CreateSubirMotor','CreateOnlineMotor','CreateSubirTablero','CreateOnlineTablero'),
                'users' => array('@'),
            ),
            array('allow',
                'actions' => array('admin', 'delete'),
                'users' => array('admin'),
            ),
            array('deny',
                'users' => array('*'),
            ),
        );
    }

    public function actionView() {
        $this->render('view', array(
            'model' => $this->loadModel(),
        ));
    }

    public function actionCreate() {
        $model = new Documentos;

        $this->performAjaxValidation($model);

        if (isset($_POST['Documentos'])) {
            $model->attributes = $_POST['Documentos'];


            if ($model->save())
                $this->redirect(array('view', 'id' => $model->id));
        }

        $this->render('create', array(
            'model' => $model,
        ));
    }

    public function actionCreateSubir() {
        $model = new Documentos;
        $modelMeta = new MetaDocs;
        $modelArchivo = new Archivos;
        
        $this->performAjaxValidation(array($model, $modelMeta, $modelArchivo));

        if (isset($_POST['Documentos'], $_POST['MetaDocs'], $_POST['Archivos'])) {
            $model->attributes = $_POST['Documentos'];
            $modelMeta->attributes = $_POST['MetaDocs'];
            $modelArchivo->attributes = $_POST['MetaDocs'];

            $valid = $model->validate();
            $valid = $modelMeta->validate() && $valid;
            $valid = $modelArchivo->validate() && $valid;


            if ($valid) {
                // guarda modelo de archivos y ys están disponible el nombre, tipo  tamaño
                
                //$modelArchivo->beforeSave();
                $modelArchivo->save(false);
                // guarda modelo de documento y ya está disponible $model->id
                $model->save(false);
                // asigna a metadoc doc el id de el modelo de documento
                $modelMeta->documento=$model->id;
                // asigna el archivo al campo ruta de metadoc
                $modelMeta->ruta=$modelArchivo->id;
                //guarda modelo de metadoc
                $modelMeta->save(false);

                $this->redirect(array('view', 'id' => $model->id));
            }
        }

        $this->render('createSubir', array(
            'model' => $model,
            'modelMeta' => $modelMeta,
            'modelArchivo' => $modelArchivo,
        ));
    }
    
    public function actionCreateSubirEquipo() {
        $model = new Documentos;
        $modelMeta = new MetaDocs;
        $modelArchivo = new Archivos;
        
        $this->performAjaxValidation(array($model, $modelMeta, $modelArchivo));

        if (isset($_POST['Documentos'], $_POST['MetaDocs'], $_POST['Archivos'])) {
            $model->attributes = $_POST['Documentos'];
            $modelMeta->attributes = $_POST['MetaDocs'];
            $modelArchivo->attributes = $_POST['Archivos'];

            $valid = $model->validate();
            $valid = $modelMeta->validate() && $valid;
            //$valid = $modelArchivo->validate() && $valid;


            if ($valid) {
                // guarda modelo de archivos y ys están disponible el nombre, tipo  tamaño
                
                //$modelArchivo->beforeSave();
                $modelArchivo->save(false);
                // guarda modelo de documento y ya está disponible $model->id
                $model->save(false);
                // asigna a metadoc doc el id de el modelo de documento
                $modelMeta->documento=$model->id;
                // asigna el archivo al campo ruta de metadoc
                $modelMeta->ruta=$modelArchivo->id;
                //guarda modelo de metadoc
                $modelMeta->save(false);

                $this->redirect(array('/metaDocs/view','id' => $modelMeta->id));
            }
        }

        $this->render('createSubirEquipo', array(
            'model' => $model,
            'modelMeta' => $modelMeta,
            'modelArchivo' => $modelArchivo,
        ));
    }
    
    public function actionCreateSubirMotor() {
        $model = new Documentos;
        $modelMeta = new MetaDocs;
        $modelArchivo = new Archivos;
        
        $this->performAjaxValidation(array($model, $modelMeta, $modelArchivo));

        if (isset($_POST['Documentos'], $_POST['MetaDocs'], $_POST['Archivos'])) {
            $model->attributes = $_POST['Documentos'];
            $modelMeta->attributes = $_POST['MetaDocs'];
            $modelArchivo->attributes = $_POST['Archivos'];

            $valid = $model->validate();
            $valid = $modelMeta->validate() && $valid;
            $valid = $modelArchivo->validate() && $valid;


            if ($valid) {
                // guarda modelo de archivos y ys están disponible el nombre, tipo  tamaño
                
                //$modelArchivo->beforeSave();
                $modelArchivo->save(false);
                // guarda modelo de documento y ya está disponible $model->id
                $model->save(false);
                // asigna a metadoc doc el id de el modelo de documento
                $modelMeta->documento=$model->id;
                // asigna el archivo al campo ruta de metadoc
                $modelMeta->ruta=$modelArchivo->id;
                //guarda modelo de metadoc
                $modelMeta->save(false);

                $this->redirect(array('/metaDocs/view','id' => $modelMeta->id));
            }
        }

        $this->render('createSubirMotor', array(
            'model' => $model,
            'modelMeta' => $modelMeta,
            'modelArchivo' => $modelArchivo,
        ));
    }
    
    public function actionCreateSubirTablero() {
        $model = new Documentos;
        $modelMeta = new MetaDocs;
        $modelArchivo = new Archivos;
        
        $this->performAjaxValidation(array($model, $modelMeta, $modelArchivo));

        if (isset($_POST['Documentos'], $_POST['MetaDocs'], $_POST['Archivos'])) {
            $model->attributes = $_POST['Documentos'];
            $modelMeta->attributes = $_POST['MetaDocs'];
            $modelArchivo->attributes = $_POST['Archivos'];

            $valid = $model->validate();
            $valid = $modelMeta->validate() && $valid;
            $valid = $modelArchivo->validate() && $valid;


            if ($valid) {
                // guarda modelo de archivos y ys están disponible el nombre, tipo  tamaño
                
                //$modelArchivo->beforeSave();
                $modelArchivo->save(false);
                // guarda modelo de documento y ya está disponible $model->id
                $model->save(false);
                // asigna a metadoc doc el id de el modelo de documento
                $modelMeta->documento=$model->id;
                // asigna el archivo al campo ruta de metadoc
                $modelMeta->ruta=$modelArchivo->id;
                //guarda modelo de metadoc
                $modelMeta->save(false);

                $this->redirect(array('/metaDocs/view','id' => $modelMeta->id));
            }
        }

        $this->render('createSubirTablero', array(
            'model' => $model,
            'modelMeta' => $modelMeta,
            'modelArchivo' => $modelArchivo,
        ));
    }
    
     public function actionCreateFisico() {
        $model = new Documentos;
        $modelMeta = new MetaDocs;

        $this->performAjaxValidation(array($model, $modelMeta));

        if (isset($_POST['Documentos'], $_POST['MetaDocs'])) {
            $model->attributes = $_POST['Documentos'];
            $modelMeta->attributes = $_POST['MetaDocs'];

            $valid = $model->validate();
            $valid = $modelMeta->validate() && $valid;


            if ($valid) {
                // guarda modelo de documento y ya está disponible $model->id
                $model->save(false);
                // asigna a metadoc doc el id de el modelo de documento
                $modelMeta->documento=$model->id;
                //guarda modelo de metadoc
                $modelMeta->save(false);

                $this->redirect(array('/metaDocs/view','id' => $modelMeta->id));
            }
        }

        $this->render('createFisico', array(
            'model' => $model,
            'modelMeta' => $modelMeta,
        ));
    }
    
     public function actionCreateOnlineEquipo() {
        $model = new Documentos;
        $modelMeta = new MetaDocs;
        $modelA = new Anotaciones;
        
        $this->performAjaxValidation(array($model, $modelMeta));

        if (isset($_POST['Documentos'], $_POST['MetaDocs'],$_POST['Anotaciones'])) {
            $model->attributes = $_POST['Documentos'];
            $modelMeta->attributes = $_POST['MetaDocs'];
            $modelA->attributes = $_POST['Anotaciones'];

            $valid = $model->validate();
            $valid = $modelMeta->validate() && $valid;
            $valid = $modelA->validate() && $valid;

            if ($valid) {
                // guarda modelo de documento y ya está disponible $model->id
                $model->save(false);
                // asigna a metadoc doc el id de el modelo de documento
                $modelMeta->documento=$model->id;
                //guarda modelo de metadoc
                $modelMeta->save(false);
                //guarda modelo de anotación
                
                //TODO: ARREGLAR EL id del usuario
                $modelA->usuario=1;
                $modelA->documento=$modelMeta->id; // ES EL ID DE METADOC no de doc.
                $modelA->descripcion=$model->descripcion;
                $modelA->save(false);

                $this->redirect(array('/metaDocs/view','id' => $modelMeta->id));
            }
        }

        $this->render('createOnlineEquipo', array(
            'model' => $model,
            'modelMeta' => $modelMeta,
            'modelA' => $modelA,
        ));
    }

     public function actionCreateOnlineMotor() {
        $model = new Documentos;
        $modelMeta = new MetaDocs;
        $modelA = new Anotaciones;
        
        $this->performAjaxValidation(array($model, $modelMeta));

        if (isset($_POST['Documentos'], $_POST['MetaDocs'],$_POST['Anotaciones'])) {
            $model->attributes = $_POST['Documentos'];
            $modelMeta->attributes = $_POST['MetaDocs'];
            $modelA->attributes = $_POST['Anotaciones'];

            $valid = $model->validate();
            $valid = $modelMeta->validate() && $valid;
            $valid = $modelA->validate() && $valid;

            if ($valid) {
                // guarda modelo de documento y ya está disponible $model->id
                $model->save(false);
                // asigna a metadoc doc el id de el modelo de documento
                $modelMeta->documento=$model->id;
                //guarda modelo de metadoc
                $modelMeta->save(false);
                //guarda modelo de anotación
                
                //TODO: ARREGLAR EL id del usuario
                $modelA->usuario=1;
                $modelA->documento=$modelMeta->id; // ES EL ID DE METADOC no de doc.
                $modelA->descripcion=$model->descripcion;
                $modelA->save(false);

                $this->redirect(array('/metaDocs/view','id' => $modelMeta->id));
            }
        }

        $this->render('createOnlineMotor', array(
            'model' => $model,
            'modelMeta' => $modelMeta,
            'modelA' => $modelA,
        ));
    }

     public function actionCreateOnlineTablero() {
        $model = new Documentos;
        $modelMeta = new MetaDocs;
        $modelA = new Anotaciones;
        
        $this->performAjaxValidation(array($model, $modelMeta));

        if (isset($_POST['Documentos'], $_POST['MetaDocs'],$_POST['Anotaciones'])) {
            $model->attributes = $_POST['Documentos'];
            $modelMeta->attributes = $_POST['MetaDocs'];
            $modelA->attributes = $_POST['Anotaciones'];

            $valid = $model->validate();
            $valid = $modelMeta->validate() && $valid;
            $valid = $modelA->validate() && $valid;

            if ($valid) {
                // guarda modelo de documento y ya está disponible $model->id
                $model->save(false);
                // asigna a metadoc doc el id de el modelo de documento
                $modelMeta->documento=$model->id;
                //guarda modelo de metadoc
                $modelMeta->save(false);
                //guarda modelo de anotación
                
                //TODO: ARREGLAR EL id del usuario
                $modelA->usuario=1;
                $modelA->documento=$modelMeta->id; // ES EL ID DE METADOC no de doc.
                $modelA->descripcion=$model->descripcion;
                $modelA->save(false);

                $this->redirect(array('/metaDocs/view','id' => $modelMeta->id));
            }
        }

        $this->render('createOnlineTablero', array(
            'model' => $model,
            'modelMeta' => $modelMeta,
            'modelA' => $modelA,
        ));
    }

   
    public function actionUpdate() {
        $model = $this->loadModel();

        $this->performAjaxValidation($model);

        if (isset($_POST['Documentos'])) {
            $model->attributes = $_POST['Documentos'];

            if ($model->save())
                $this->redirect(array('view', 'id' => $model->id));
        }

        $this->render('update', array(
            'model' => $model,
        ));
    }

    public function actionDelete() {
        if (Yii::app()->request->isPostRequest) {
            $this->loadModel()->delete();

            if (!isset($_GET['ajax']))
                $this->redirect(array('index'));
        }
        else
            throw new CHttpException(400,
                    Yii::t('app', 'Invalid request. Please do not repeat this request again.'));
    }

    public function actionIndex() {
        $dataProvider = new CActiveDataProvider('Documentos');
        $this->render('index', array(
            'dataProvider' => $dataProvider,
        ));
    }

    public function actionAdmin() {
        $model = new Documentos('search');
        if (isset($_GET['Documentos']))
            $model->attributes = $_GET['Documentos'];

        $this->render('admin', array(
            'model' => $model,
        ));
    }

    public function loadModel() {
        if ($this->_model === null) {
            if (isset($_GET['id']))
                $this->_model = Documentos::model()->findbyPk($_GET['id']);
            if ($this->_model === null)
                throw new CHttpException(404, Yii::t('app', 'The requested page does not exist.'));
        }
        return $this->_model;
    }

    protected function performAjaxValidation($models) {
        if (isset($_POST['ajax']) && $_POST['ajax'] === 'documentos-form') {
            echo CActiveForm::validate($models);
            Yii::app()->end();
        }
    }

}
