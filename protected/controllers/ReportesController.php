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
// funciÃƒÂ³n para marcar como arreglado el ÃƒÂºltimo aviso de un modelo.
function arreglarUltimo($modelIn) {
    $equipo=Estructura::model()->findBySql('select * from estructura where Equipo="' . $modelIn->Equipo . '"');
    if (!isset($equipo)) {
        echo '<script language="JavaScript">
                        alert("El equipo no existe");
                     </script>';
        return(0);
    }
    if (!isset($equipo->plan_mant_ultrasonido)) {
        echo '<script language="JavaScript">
                        alert("No se generÃƒÂ³ aviso ZI para esta mediciÃƒÂ³n porque el equipo no tiene plan de mantenimiento de ultrasonido, por favor ingrese el plan de mantenimiento.");
                     </script>';
        return(0);
    }
    if ($equipo->plan_mant_ultrasonido == 0) {
        echo '<script language="JavaScript">
                        alert("No se generÃƒÂ³ aviso ZI para esta mediciÃƒÂ³n porque el equipo no tiene plan de mantenimiento de ultrasonido, por favor ingrese el plan de mantenimiento.");
                    </script>';
        return(0);
    }
    $modelAviso = AvisosZI::model()->findBySql('select * from avisosZI where plan_mant=' . $equipo->plan_mant_ultrasonido . ' AND arreglado=0 order by Fecha desc');

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
    $modelAviso->Ruta = "/index.php/reportes/" . $modelIn->id;
    $modelAviso->Operador = Yii::app()->user->name;
    $modelAviso->Estado = $modelIn->Estado;
    $modelAviso->Observaciones = $modelIn->Observaciones;
    $modelAviso->OT = $modelIn->OT;
    $equipo = Estructura::model()->findBySql('select * from estructura where Equipo="' . $modelIn->Equipo . '"');
    //busca el cÃƒÂ³digo del equipo al que pertenece el motor
    if (!isset($equipo)) {
        echo '<script language="JavaScript">
                        alert("El equipo no existe");
                     </script>';
        return(0);
    }
    $modelAviso->Codigo = $equipo->Codigo;
    if (!isset($equipo->plan_mant_ultrasonido)) {
        echo '<script language="JavaScript">
                        alert("No se generÃƒÂ³ aviso ZI para esta mediciÃƒÂ³n porque el equipo no tiene plan de mantenimiento, por favor ingrese el plan de mantenimiento del equipo.");
                     </script>';
        return(0);
    }
    if ($equipo->plan_mant_ultrasonido == 0) {
        echo '<script language="JavaScript">
                        alert("No se generÃƒÂ³ aviso ZI para esta mediciÃƒÂ³n porque el equipo no tiene plan de mantenimiento, por favor ingrese el plan de mantenimiento del equipo.");
                     </script>';
        return(0);
    }
    $modelAviso->plan_mant = $equipo->plan_mant_ultrasonido;
    if (!$modelAviso->save())
        echo '<script language="JavaScript">
                            alert("Error guardando el aviso ZI.");
                        </script>';
    return(1);
}

class ReportesController extends Controller {

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
                'actions' => array('DynamicReportes', 'DynamicReportesArea'),
                'users' => array('*'),
            ),
            array('allow',
                'actions' => array('index', 'view', 'update','admin', 'create', 'passthru'),
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

    public function actionPassthru() {
        //$file = urldecode($_GET['path']);
        $file = $_GET['path'];

        $fallback = 'c:\\wamp\\www\\images\\fallback.gif';

//DETERMINE TYPE
        $ext = array_pop(explode('.', $file));
        $image = array_pop(explode('\\', $file));
        $allowed['gif'] = 'image/gif';
        $allowed['png'] = 'image/png';
        $allowed['jpg'] = 'image/jpeg';
        $allowed['jpeg'] = 'image/jpeg';
        $allowed['pdf'] = 'application/pdf';

        if (file_exists($file) && $ext != '' && isset($allowed[strToLower($ext)])) {
            $type = $allowed[strToLower($ext)];
        } else {
            $file = $fallback;
            $type = 'image/gif';
        }

//header("Content-type: {$type}Content-Disposition: inline; filename=\"{$image}\"Content-length: ".(string)(filesize($file))); 
        header('Pragma: public');
        header('Expires: 0');
        header('Cache-Control: must-revalidate, post-check=0, pre-check=0');
        header('Content-Transfer-Encoding: binary');
        header('Content-length: ' . filesize($file));
        header('Content-Type: ' . $type);
        header('Content-Disposition: attachment; filename=' . $image);

        @readfile($file);
        exit();
    }

    public function actionView() {
        $this->render('view', array(
            'model' => $this->loadModel(),
        ));
    }

    public function actionCreate() {
        $model = new Reportes;
        $modelArchivo = new Archivos;
                
        $this->performAjaxValidation(array($model,$modelArchivo));
        $continuar = 1;
                
        if (isset($_POST['Reportes'])) {
            $model->attributes = $_POST['Reportes'];
            $modelArchivo->nombre=$model->Path;
            $modelArchivo->save(false);
            $model->Path="".$modelArchivo->id;
            if ($model->save()) {
                 
                 crearEvento('ULTRASONIDO','CREAR','Medición creada');
                                
                $result2 = 1;
                $result1 = arreglarUltimo($model);
                if ($model->Estado < 3)
                    $result2 = crearAviso($model);
                if (($result1 == 1) && ($result2 == 1))
                    $this->redirect(array('view', 'id' => $model->id));
                else {
                            $equipo=Estructura::model()->findBySql('select * from estructura where Equipo="'.$model->Equipo.'"');
                                    if (isset($equipo))
                                        echo '<a href=/index.php/estructura/update?id='.$equipo->id.'">á é í ó ú  Haga click aquí para ingresar el plan de mantenimiento del equipo '.$equipo->Equipo.'</a>';
                                    $continuar=1;                }
            }
        }
        if (isset($_GET['id'])) {
            $equipo = Estructura::model()->findBySql('select * from estructura where Equipo="' . $_GET['id'] . '"');
            if (isset($equipo)) {
                if (!isset($equipo->plan_mant_ultrasonido)) {
                    // se setea el plan_mant_aislamiento según la nomenclatura:
                    // vib=+11000000,ais=12000000,lub=13000000,ultra=14000000,termo=15000000
                    $motor->plan_mant_ultrasonido=$motor->id+14000000;
                    // se guarda el modelo
                    $motor->save();
                    $continuar = 1;
                }else 
                if ($equipo->plan_mant_ultrasonido == 0) {
                    // se setea el plan_mant_aislamiento según la nomenclatura:
                    // vib=+11000000,ais=12000000,lub=13000000,ultra=14000000,termo=15000000
                    $motor->plan_mant_ultrasonido=$motor->id+14000000;
                    // se guarda el modelo
                    $motor->save();
                    $continuar = 1;
                }
            }
        }    
        if ($continuar == 1) {
            $this->render('create', array(
                'model' => $model,
                'modelArchivo' => $modelArchivo,
            ));
        }
    }

    public function actionUpdate() {
        $model = $this->loadModel();
        if (is_numeric($model->Path))
            $modelArchivo=  Archivos::model()->findbysql("select * from archivos where id=".$model->Path);
        else
            $modelArchivo=new Archivos();
        
        $this->performAjaxValidation($model);

        if (isset($_POST['Reportes'])) {
            $model->attributes = $_POST['Reportes'];
            $modelArchivo->nombre = $model->Path;
            $modelArchivo->save(false);
            $model->Path = "" . $modelArchivo->id;
            if ($model->save())
            {
                
                                crearEvento('ULTRASONIDO','ACTUALIZAR','Medición actualizada');
                $this->redirect(array('view', 'id' => $model->id));
            }
        }

        $this->render('update', array(
            'model' => $model,
            'modelArchivo' => $modelArchivo,
        ));
    }

    public function actionDelete() {
        if (Yii::app()->request->isPostRequest) {
            $this->loadModel()->delete();
crearEvento('ULTRASONIDO','BORRAR','Medición borrada');
            if (!isset($_GET['ajax']))
                $this->redirect(array('index'));
        }
        else
            throw new CHttpException(400,
                    Yii::t('app', 'Invalid request. Please do not repeat this request again.'));
    }

    public function actionIndex() {
        $dataProvider = new CActiveDataProvider('Reportes');
        $this->render('index', array(
            'dataProvider' => $dataProvider,
        ));
    }

    public function actionAdmin() {
        $model = new Reportes('search');
        if (isset($_GET['Reportes']))
            $model->attributes = $_GET['Reportes'];

        $this->render('admin', array(
            'model' => $model,
        ));
    }

    public function loadModel() {
        if ($this->_model === null) {
            if (isset($_GET['id']))
                $this->_model = Reportes::model()->findbyPk($_GET['id']);
            if ($this->_model === null)
                throw new CHttpException(404, Yii::t('app', 'The requested page does not exist.'));
        }
        return $this->_model;
    }

    protected function performAjaxValidation($model) {
        if (isset($_POST['ajax']) && $_POST['ajax'] === 'reportes-form') {
            echo CActiveForm::validate($model);
            Yii::app()->end();
        }
    }

    public function actionDynamicReportes() {

        $data = array();
        $data["area"] = $_GET['area'];
        $data["equipo"] = $_GET['equipo'];
        if ($_GET['equipo'] == "")
            $data["equipo"] = "ZZZZXXXXDDEE";
        echo $this->renderPartial('_adminGridView', $data, true, true);
        //echo "HERE3";
    }

    public function actionDynamicReportesArea() {
        $data = array();
        $data["area"] = $_GET['area'];

        if ($_GET['area'] == "")
            $data["area"] = "ZZZZXXXXDDEE";
        echo $this->renderPartial('_adminGridViewArea', $data, true, true);
        //echo "HERE4";
    }

}
