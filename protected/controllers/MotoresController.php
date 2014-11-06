<?php
    function getRealIpAddr()
    {
        if (!empty($_SERVER['HTTP_CLIENT_IP']))   //check ip from share internet
        {
        $ip=$_SERVER['HTTP_CLIENT_IP'];
        }
        elseif (!empty($_SERVER['HTTP_X_FORWARDED_FOR']))   //to check ip is pass from proxy
        {
        $ip=$_SERVER['HTTP_X_FORWARDED_FOR'];
        }
        else
        {
        $ip=$_SERVER['REMOTE_ADDR'];
        }
        return $ip;
    }
    function crearEvento($modulo,$operacion,$descripcion)
    {
        
        $evento=  new Eventos;
        if ($operacion=="LOGIN_FAIL") $evento->usuario="";
        else $evento->usuario=Yii::app()->user->name;
        $evento->modulo=$modulo;
        $evento->operacion=$operacion;
        $evento->ip=getRealIpAddr();
        $evento->descripcion=$descripcion;
        $evento->fecha=date("Y-m-d H:i:s");
        $evento->save();
    }

    class MotoresController extends Controller
    {
        public $layout = '//layouts/responsiveLayout';
        private $_model;

        public function filters()
        {
            return array (
                'accessControl',
            );
        }

        public function accessRules()
        {
            return array (
                array ('allow',
                    'actions' => array ('index', 'view', 'dynamicArea', 'dynamicEquipo', 'dynamicEquipoVacio','dynamicMotores','dynamicFMotor','tituloSearch'),
                    'users' => array ('*'),
                ),
                array ('allow',
                    'actions' => array ('update','admin', 'create', 'updatePM'),
                    'users' => array ('@'),
                ),
                array ('allow',
                    'actions' => array ('delete','updatePM'),
                    'users' => array ('admin'),
                ),
                array ('deny',
                    'users' => array ('*'),
                ),
            );
        }

        public function actionView()
        {
            $this->render('view',
                          array (
                'model' => $this->loadModel(),
            ));
        }

        public function actionCreate()
        {
            $model = new Motores;
 $modelArchivo = new Archivos;
            $this->performAjaxValidation($model);

            if (isset($_POST['Motores']))
            {
                $model->attributes = $_POST['Motores'];
                $modelArchivo->nombre=$model->PathFoto;
                $modelArchivo->save(false);
                $model->PathFoto="".$modelArchivo->id;

                if ($model->save())
                {
                                                    crearEvento('MOTORES','CREAR','Motor creado');

                    $this->redirect(array ('view', 'id' => $model->id));
                }
            }

            $this->render('create',
                          array (
                'model' => $model,
                'modelArchivo' => $modelArchivo,
            ));
        }

        public function actionUpdate()
        {
            $model = $this->loadModel();
 if (is_numeric($model->PathFoto))
            $modelArchivo=  Archivos::model()->findbysql("select * from archivos where id=".$model->PathFoto);
        else
            $modelArchivo=new Archivos();
            $this->performAjaxValidation($model);

            if (isset($_POST['Motores']))
            {
                $model->attributes = $_POST['Motores'];
            $modelArchivo->nombre = $model->PathFoto;
            $modelArchivo->save(false);
            $model->PathFoto = "" . $modelArchivo->id;
                if ($model->save()){
                                                    crearEvento('MOTORES','ACTUALIZAR','Motor actualizado');

                    $this->redirect('/index.php/site/inicio');
                    
                }
            }

            $this->render('update',
                          array (
                'model' => $model,
            'modelArchivo' => $modelArchivo,
            ));
        }

        public function actionUpdatePM()
        {
            $model = $this->loadModel();

            $this->performAjaxValidation($model);

            if (isset($_POST['Motores']))
            {
                $model->attributes = $_POST['Motores'];

                if ($model->save()){
                    $this->redirect(array ('view', 'id' => $model->id));
                }
            }

            $this->render('updatePM',
                          array (
                'model' => $model,
            ));
        }

        public function actionDelete()
        {
            if (Yii::app()->request->isPostRequest)
            {
                $this->loadModel()->delete();
                                crearEvento('MOTORES','BORRAR','Motor borrado');                    

                if (!isset($_GET['ajax']))
                    $this->redirect(array ('index'));
            }
            else
                throw new CHttpException(400,
                    Yii::t('app',
                           'Invalid request. Please do not repeat this request again.'));
        }

        public function actionIndex()
        {
            $dataProvider = new CActiveDataProvider('Motores');
            $this->render('index',
                          array (
                'dataProvider' => $dataProvider,
            ));
        }

        public function actionAdmin()
        {
            $model = new Motores('search');
            if (isset($_GET['Motores']))
                $model->attributes = $_GET['Motores'];

            $this->render('admin',
                          array (
                'model' => $model,
            ));
        }

        public function loadModel()
        {
            if ($this->_model === null)
            {
                if (isset($_GET['id']))
                    $this->_model = Motores::model()->findbyPk($_GET['id']);
                if ($this->_model === null)
                    throw new CHttpException(404, Yii::t('app',
                                                         'The requested page does not exist.'));
            }
            return $this->_model;
        }

        protected function performAjaxValidation($model)
        {
            if (isset($_POST['ajax']) && $_POST['ajax'] === 'motores-form')
            {
                echo CActiveForm::validate($model);
                Yii::app()->end();
            }
        }

        public function actionDynamicArea()
        {
            $query = 'WHERE Proceso="' . $_GET['proceso'];
            $data = CHtml::listData(Estructura::model()->findAllbySql(
                        'SELECT DISTINCT Area FROM estructura ' . $query . '" ORDER BY Area'),
                        'Area',
                        'Area'
            );
            //en value 1 Areas que pertenecen al proceso.
            $value1 = "";
            $value1 = $value1 . CHtml::tag('option',
                                               array ('value' => ""),
                                               CHtml::encode('Por favor seleccione el proceso'),
                                                             true);
            foreach ($data as $value => $name)
            {
                $value1 = $value1 . CHtml::tag('option',
                                               array ('value' => $value),
                                               CHtml::encode($name),
                                                             true);
            }
            $response = array ('value1' => $value1);
            echo CJSON::encode($response);
        }

        public function actionDynamicEquipo()
        {
            $query = 'WHERE Area="' . $_GET['area'];
            $data = CHtml::listData(Estructura::model()->findAllbySql(
                        'SELECT Equipo FROM estructura ' . $query . '" ORDER BY Equipo'),
                        'Equipo',
                        'Equipo'
            );
            //en value 1 Areas que pertenecen al proceso.
            $value1 = "";
            $value1 = $value1 . CHtml::tag('option',
                                               array ('value' => ""),
                                               CHtml::encode('Por favor seleccione el equipo'),
                                                             true);
            foreach ($data as $value => $name)
            {
                $value1 = $value1 . CHtml::tag('option',
                                               array ('value' => $value),
                                               CHtml::encode($name),
                                                             true);
            }
            $response = array ('value1' => $value1);
            echo CJSON::encode($response);
        }  
        
        public function actionDynamicEquipoVacio()
        {
            
            $value1 = "";
            $value1 = $value1 . CHtml::tag('option',
                                               array ('value' => 0),
                                               CHtml::encode('Por favor seleccione un equipo para filtrar los resultados'),
                                                             true);
            
            $response = array ('value1' => $value1);
            echo CJSON::encode($response);
        }          
          
        
        public function actionDynamicFMotor()
        {
            $query = 'WHERE Equipo="' . $_GET['equipo'];
            $data = CHtml::listData(Motores::model()->findAllbySql(
                        'SELECT TAG, CONCAT(TAG," - ",Motor) as Motor FROM motores ' . $query . '" ORDER BY TAG'),
                        'TAG',
                        'Motor'
            );
            //en value 1 Areas que pertenecen al proceso.
            $value1 = CHtml::tag('option',
                                               array ('value' => 0),
                                               CHtml::encode("Por favor seleccione un motor."),
                                                             true);;
            foreach ($data as $value => $name)
            {
                $value1 = $value1 . CHtml::tag('option',
                                               array ('value' => $value),
                                               CHtml::encode($name),
                                                             true);
            }
            $response = array ('value1' => $value1);
            echo CJSON::encode($response);
        }   

        public function actionDynamicMotores()
        {
            $data = array();
            $data["area"] = $_GET['area'];
            $data["equipo"] = $_GET['equipo'];
            $this->renderPartial('_adminGridView', $data, false, true);
        }
        
        public function actionTituloSearch($term) {
            // TODO: Arreglar para que busque también cuando se tecleé el nombre, si conviene.
            //$sql = 'SELECT docIdent FROM clientes WHERE docIdent LIKE \'%' . trim($term) . '%\' LIMIT 10';
            $sql = '(SELECT CONCAT(TAG,"-",Equipo,"-",Motor) as label, TAG as value FROM motores WHERE TAG LIKE :qterm OR Equipo LIKE :qterm OR Motor LIKE :qterm ORDER BY Motor ASC LIMIT 8)
                    UNION (SELECT CONCAT(TAG,"-",Area,"-",Tablero) as label, TAG as value FROM tableros WHERE TAG LIKE :qterm OR Area LIKE :qterm OR Tablero LIKE :qterm ORDER BY Tablero ASC LIMIT 5)
                    UNION (SELECT CONCAT(Codigo,"-",Area,"-",Equipo) as label, Codigo as value FROM estructura WHERE Codigo LIKE :qterm OR Area LIKE :qterm OR Equipo LIKE :qterm ORDER BY Equipo ASC LIMIT 5)';
    //        $sql = 'SELECT docIdent FROM clientes WHERE docIdent LIKE \'%94%\' LIMIT 10';
            $command = Yii::app()->db->createCommand($sql);
            $qterm = '%' . $_GET['term'] . '%';
            $command->bindParam(":qterm", $qterm, PDO::PARAM_STR);
            $result = $command->queryAll();
            echo CJSON::encode($result);
            exit;
        }
        
}
    