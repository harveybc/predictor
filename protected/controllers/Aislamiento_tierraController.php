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
// función para marcar como arreglado el Ãºltimo aviso de un modelo.
function arreglarUltimo($modelIn) {
    $motor = Motores::model()->findBySql('select * from motores where TAG="' . $modelIn->TAG . '"');
    if (!isset($motor)) {
        echo '<script language="JavaScript">
                        alert("El motor no existe");
                     </script>';
        return(0);
    }
    if (!isset($motor->plan_mant_aislamiento)) {
        echo '<script language="JavaScript">
                        alert("Atención: NO se generó aviso ZI para esta medición porque el motor no tiene plan de mantenimiento de aislamiento, por favor ingrese el plan de mantenimiento..");
                     </script>';
        return(0);
    }
    if ($motor->plan_mant_aislamiento == 0) {
        echo '<script language="JavaScript">
                        alert("Atención: NO se generó aviso ZI para esta medición porque el motor no tiene plan de mantenimiento de aislamiento, por favor ingrese el plan de mantenimiento..");
                    </script>';
        return(0);
    }
    $modelAviso = AvisosZI::model()->findBySql('select * from avisosZI where plan_mant=' . $motor->plan_mant_aislamiento . ' AND arreglado=0 order by Fecha desc');

    if (isset($modelAviso)) {
        $modelAviso->arreglado = 1;
        if (!$modelAviso->save())
            echo '<script language="JavaScript">
                            alert("Error guardando el aviso ZI.");
                        </script>';
    }
    return(1);
}

function crearAviso($modelIn) {
    $modelAviso = new AvisosZI;
    $modelAviso->arreglado = 0;
    $modelAviso->Ruta = "/index.php/aislamiento_tierra/" . $modelIn->Toma;
    $modelAviso->Operador = Yii::app()->user->name;
    $modelAviso->Estado = $modelIn->Estado;
    $modelAviso->Observaciones = $modelIn->Observaciones;
    $modelAviso->OT = $modelIn->OT;
    $motor = Motores::model()->findBySql('select * from motores where TAG="' . $modelIn->TAG . '"');
    //busca el código del equipo al que pertenece el motor
    if (!isset($motor)) {
        echo '<script language="JavaScript">
                        alert("El motor no existe");
                     </script>';
        return(0);
    }
    $equipo = Estructura::model()->findBySql('select * from estructura where Equipo="' . $motor->Equipo . '"');
    if (!isset($equipo)) {
        echo '<script language="JavaScript">
                        alert("El equipo al que pertenece el motor no se pudo encontrar");
                     </script>';
        return(0);
    }
    $modelAviso->Codigo = $equipo->Codigo;
    if (!isset($motor->plan_mant_aislamiento)) {
        echo '<script language="JavaScript">
                        alert("Atención: NO se generó aviso ZI para esta medición porque el motor no tiene plan de mantenimiento, por favor ingrese el plan de mantenimiento.");
                     </script>';
        return(0);
    }
    if ($motor->plan_mant_aislamiento == 0) {
        echo '<script language="JavaScript">
                        alert("Atención: NO se generó aviso ZI para esta medición porque el motor no tiene plan de mantenimiento, por favor ingrese el plan de mantenimiento.");
                     </script>';
        return(0);
    }
    $modelAviso->plan_mant = $motor->plan_mant_aislamiento;
    if (!$modelAviso->save())
        echo '<script language="JavaScript">
                            alert("Error guardando el aviso ZI.");
                        </script>';
    return(1);
}

class Aislamiento_tierraController extends Controller {

    public $layout='//layouts/responsiveLayout'; 
    private $_model;

    public function filters() {
        return array(
            'accessControl',
        );
    }

    public function accessRules() {
        return array(
            array('allow',
                'actions' => array('dynamicFechas', 'graphFase'),
                'users' => array('*'),
            ),
            array('allow',
                'actions' => array('index', 'view', 'update','create', 'admin'),
                'users' => array('@'),
            ),
            array('allow',
                'actions' => array('delete'),
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
        $model = new Aislamiento_tierra;
        $this->performAjaxValidation($model);
        $continuar = 1;
        if (isset($_POST['Aislamiento_tierra'])) {
            $model->attributes = $_POST['Aislamiento_tierra'];
            if ($model->save()) {
                                                crearEvento('AISLAMIENTO','CREAR','Medición creada');
                              

                $result2 = 1;
                $result1 = arreglarUltimo($model);
                if ($model->Estado < 3)
                    $result2 = crearAviso($model);
                if (($result1 == 1) && ($result2 == 1))
                    {

                    $this->redirect(array('view', 'Toma' => $model->Toma));
                    }
                else {
                    $motor=Motores::model()->findBySql('select * from motores where TAG="'.$model->TAG.'"');
                                    if (isset($motor))
                                        echo '<a href=/index.php/motores/update?id='.$motor->id.'">Haga click aquí para ingresar el plan de mantenimiento del motor .'.$motor->Motor.'</a>';
                                    $continuar=1;
                }
            }
        }
        
        if (isset($_GET['id'])) {
            $motor = Motores::model()->findBySql('select * from motores where TAG="' . $_GET['id'] . '"');
            if (isset($motor)) {
                if (!isset($motor->plan_mant_aislamiento)) {
                    // se setea el plan_mant_aislamiento según la nomenclatura:
                    // vib=+11000000,ais=12000000,lub=13000000,ultra=14000000,termo=15000000
                    $motor->plan_mant_aislamiento=$motor->id+12000000;
                    // se guarda el modelo
                    $motor->save();
                    $continuar = 1;
                }else 
                if ($motor->plan_mant_aislamiento == 0) {
                    // se setea el plan_mant_aislamiento según la nomenclatura:
                    // vib=+11000000,ais=12000000,lub=13000000,ultra=14000000,termo=15000000
                    $motor->plan_mant_aislamiento=$motor->id+12000000;
                    // se guarda el modelo
                    $motor->save();
                    $continuar = 1;
                }
            }
        }        
        if ($continuar == 1) {
            $this->render('create', array(
                'model' => $model,
            ));
        }

    }

    /**
     * Grafica todos los modelos (similar a actionIndex).
     */
    public function actionGraphFase() {
        $TAG_in = $_GET['TAG'];
        if ($TAG_in == "") {
            echo "No hay datos para graficar.";
            return;
        }
        // arreglo en formato highstock
        $arrSalidaA = array();
        $arrSalidaB = array();
        $arrSalidaC = array();
        // obtiene una array de modelos que tienen el tag
        $data = Aislamiento_tierra::model()->findAllBySql("select * from aislamiento_tierra where TAG=\"" . $TAG_in . "\" order by Fecha ASC");
        // para cada uno de los modelos calcula el IP de cada fase y lo almacena en formato de stockhigh
        foreach ($data as $foreignobj) {
            // hace push en arrSalidaA,B,C de un array(strtotime($foreignobj->Fecha)*1000,$foreignobj->A10/$foreignobj->A1))
            if (($foreignobj->A1 > 0) && ($foreignobj->B1 > 0) && ($foreignobj->C1 > 0)) {
                array_push($arrSalidaA, array(strtotime($foreignobj->Fecha) * 1000, $foreignobj->A10 / $foreignobj->A1));
                array_push($arrSalidaB, array(strtotime($foreignobj->Fecha) * 1000, $foreignobj->B10 / $foreignobj->B1));
                array_push($arrSalidaC, array(strtotime($foreignobj->Fecha) * 1000, $foreignobj->C10 / $foreignobj->C1));
            } else {
                echo "Error: División por Cero (A1=0 o B1=0 o C1=0)";
                return(0);
            }
        }
        // pasa arrSalida como parÃ¡metro al renderpartial de _graph
        echo $this->renderPartial('_graph', array(
            'arrSalidaA' => $arrSalidaA,
            'arrSalidaB' => $arrSalidaB,
            'arrSalidaC' => $arrSalidaC,
            'TAG' => $TAG_in,
                ), true, true);
    }

    public function actionUpdate() {
        $model = $this->loadModel();

        $this->performAjaxValidation($model);

        if (isset($_POST['Aislamiento_tierra'])) {
            $model->attributes = $_POST['Aislamiento_tierra'];

            if ($model->save()) {
                                crearEvento('AISLAMIENTO','ACTUALIZAR','Medición actualizada');

                //crea un modelo de pendiente
                $mPend = new Pendientes;
                //llena los atributos del modelo de pendiente
                $mPend->revisado = 0;
                $mPend->fecha_enviado = date("Y-m-d");
                $mPend->ruta = "/index.php/aislamiento_tierra/update/" . $model->Toma;
                $mPend->usuario = Yii::app()->user->name;
                echo "here3";
                if ($mPend->save())
                    $this->redirect(array('/aislamiento_tierra/view', 'id' => $model->Toma));
                else
                // redirecciona a view
                    print_r($mPend->getErrors());
                //$this->redirect(array('view', 'id' => $model->Toma));
            }
        }

        $this->render('update', array(
            'model' => $model,
        ));
    }

    public function actionDelete() {
        if (Yii::app()->request->isPostRequest) {
            $this->loadModel()->delete();
                                crearEvento('AISLAMIENTO','BORRAR','Medición borrada');

            if (!isset($_GET['ajax']))
                $this->redirect(array('admin'));
        }
        else
            throw new CHttpException(400,
                    Yii::t('app', 'Invalid request. Please do not repeat this request again.'));
    }

    /**
     * @sitemap
     */
    public function actionIndex() {
        $dataProvider = new CActiveDataProvider('Aislamiento_tierra');
        $this->render('index', array(
            'dataProvider' => $dataProvider,
        ));
    }

    public function actionAdmin() {
        $model = new Aislamiento_tierra('search');
        if (isset($_GET['Aislamiento_tierra']))
            $model->attributes = $_GET['Aislamiento_tierra'];

        $this->render('admin', array(
            'model' => $model,
        ));
    }

    public function loadModel() {
        if ($this->_model === null) {
            if (isset($_GET['Toma']) || isset($_GET['id'])) {
                if (isset($_GET['Toma']))
                    $this->_model = Aislamiento_tierra::model()->findbyPk($_GET['Toma']);
                if (isset($_GET['id']))
                    $this->_model = Aislamiento_tierra::model()->findbyPk($_GET['id']);
            }
            if ($this->_model === null)
                throw new CHttpException(404, Yii::t('app', 'The requested page does not exist.'));
        }
        return $this->_model;
    }

    protected function performAjaxValidation($model) {
        if (isset($_POST['ajax']) && $_POST['ajax'] === 'aislamiento-tierra-form') {
            echo CActiveForm::validate($model);
            Yii::app()->end();
        }
    }

    public function actionDynamicFechas() {
        $data = array();
        $data["TAG"] = $_GET['TAG'];
        $this->renderPartial('_adminGridView', $data, false, true);
    }

}
