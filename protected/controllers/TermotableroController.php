<?php

function getRealIpAddr() {
    if (!empty($_SERVER['HTTP_CLIENT_IP'])) {   //check ip from share internet
        $ip = $_SERVER['HTTP_CLIENT_IP'];
    } elseif (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {   //to check ip is pass from proxy
        $ip = $_SERVER['HTTP_X_FORWARDED_FOR'];
    } else {
        $ip = $_SERVER['REMOTE_ADDR'];
    }
    return $ip;
}

function crearEvento($modulo, $operacion, $descripcion) {

    $evento = new Eventos;
    if ($operacion == "LOGIN_FAIL")
        $evento->usuario = "";
    else
        $evento->usuario = Yii::app()->user->name;
    $evento->modulo = $modulo;
    $evento->operacion = $operacion;
    $evento->ip = getRealIpAddr();
    $evento->descripcion = $descripcion;
    $evento->fecha = date("Y-m-d H:i:s");
    $evento->save();
}

// funciÃƒÂ³n para marcar como arreglado el ÃƒÂºltimo aviso de un modelo.
function arreglarUltimo($modelIn) {
    $tablero = Tableros::model()->findBySql('select * from tableros where TAG="' . $modelIn->TAG . '"');
    if (!isset($tablero)) {
        echo '<script language="JavaScript">
                        alert("El tablero no existe");
                     </script>';
        return(0);
    }
    if (!isset($tablero->plan_mant_termografia)) {
        echo '<script language="JavaScript">
                        alert("AtenciÃƒÂ³n: NO se generÃƒÂ³ aviso ZI para esta mediciÃƒÂ³n porque el tablero no tiene plan de mantenimiento de termografia, por favor ingrese el plan de mantenimiento..");
                     </script>';
        return(0);
    }
    if ($tablero->plan_mant_termografia == 0) {
        echo '<script language="JavaScript">
                        alert("AtenciÃƒÂ³n: NO se generÃƒÂ³ aviso ZI para esta mediciÃƒÂ³n porque el tablero no tiene plan de mantenimiento de termografia, por favor ingrese el plan de mantenimiento..");
                    </script>';
        return(0);
    }
    $modelAviso = AvisosZI::model()->findBySql('select * from avisosZI where plan_mant=' . $tablero->plan_mant_termografia . ' AND arreglado=0 order by Fecha desc');

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
    $modelAviso->Ruta = "/index.php/termotablero/" . $modelIn->id;
    $modelAviso->Operador = Yii::app()->user->name;
    $modelAviso->Estado = $modelIn->Estado;
    $modelAviso->Observaciones = $modelIn->Observaciones;
    $modelAviso->OT = $modelIn->OT;
    $tablero = Tableros::model()->findBySql('select * from tableros where TAG="' . $modelIn->TAG . '"');
    //busca el cÃƒÂ³digo del tablero al que pertenece el motor
    if (!isset($tablero)) {
        echo '<script language="JavaScript">
                        alert("El tablero no existe");
                     </script>';
        return(0);
    }
    $modelAviso->Codigo = $tablero->id;
    if (!isset($tablero->plan_mant_termografia)) {
        echo '<script language="JavaScript">
                        alert("AtenciÃƒÂ³n: NO se generÃƒÂ³ aviso ZI para esta mediciÃƒÂ³n porque el tablero no tiene plan de mantenimiento, por favor ingrese el plan de mantenimiento.");
                     </script>';
        return(0);
    }
    if ($tablero->plan_mant_termografia == 0) {
        echo '<script language="JavaScript">
                        alert("AtenciÃƒÂ³n: NO se generÃƒÂ³ aviso ZI para esta mediciÃƒÂ³n porque el tablero no tiene plan de mantenimiento, por favor ingrese el plan de mantenimiento.");
                     </script>';
        return(0);
    }
    $modelAviso->plan_mant = $tablero->plan_mant_termografia;
    if (!$modelAviso->save())
        echo '<script language="JavaScript">
                            alert("Error guardando el aviso ZI.");
                        </script>';
    return(1);
}

class TermotableroController extends Controller {

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
                'actions' => array('dynamicTableros', 'dynamicTableros_create', 'dynamicTAG', 'dynamicGridTAG'),
                'users' => array('*'),
            ),
            array('allow',
                'actions' => array('index', 'view', 'update', 'admin', 'create', 'guardarFile'),
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

    public function actionGuardarFile() {
        // ubicaciÃƒÂ³n de mysql y de wwwroot DEBEN terminar en "/" 
        $mysqlPath = "";
        $wwwroot = "/usr/share/test/oralx8/";
        $bduser = "harveybc";
        $bdpass = "0ptimus";
        $itp = ($bdpass == "") ? "" : "-p" . $bdpass;
        // procesa los parÃƒÂ¡metros GET 
        if (isset($_GET['action'])) {
            // 1 = crear backup completo
            if ($_GET['action'] == 1) {
                exec($mysqlPath . "mysqldump -u " . $bduser . " " . $itp . " oralxfinal >" . $wwwroot . "sqlcompleto.sql");
                header('Content-Type: application/sql');
                header('Content-Disposition: attachment; filename=BACKUP_COMPLETO_' . date("Y-m-d") . '.sql');
                header('Pragma: no-cache');
                echo readfile($wwwroot . 'sqlcompleto.sql');
            }
            // 2 = guarda el backup y lo renombra en el servidor
            if ($_GET['action'] == 2) {
                $target_path = $wwwroot . "uploads/";
                $target_path = $target_path . "uploaded.sql";
                if (move_uploaded_file($_FILES['uploadedfile']['tmp_name'], $target_path)) {
                    echo "El archivo " . basename($_FILES['uploadedfile']['name']) . " ha sido importado.<br/>";
                    // borra la bd
                    echo exec($mysqlPath . 'mysql -u ' . $bduser . ' ' . $itp . ' oralxfinal -e "DROP DATABASE IF EXISTS oralxfinal"');
                    // crea la bd CREATE DATABASE IF NOT EXISTS oralxfinal
                    echo exec($mysqlPath . 'mysql -u ' . $bduser . ' ' . $itp . ' -e "CREATE DATABASE IF NOT EXISTS oralxfinal"');
                    echo "La base de datos se ha vaciado<br/>";
                    exec($mysqlPath . 'mysql -u ' . $bduser . ' ' . $itp . ' oralxfinal < ' . $wwwroot . '/uploads/uploaded.sql');
                    echo "La base de datos se ha restaurado correctamente desde el archivo.<br/>";
                    $this->render('backup', array());
                } else {
                    echo "Hubo un error al subir el archivo!";
                }
            }
            // 3 = crear backup PARCIAL (no hace backup de seddes,examenes ni usuarios)
            if ($_GET['action'] == 3) {
                // crea el archivo a bajar
                if ($mysqlPath == "") { //para linux
                    exec('echo \'' . $mysqlPath . 'mysqlbinlog -t --read-from-remote-server --database=oralxfinal -u' . $bduser . ' ' . $itp . ' --start-datetime="' . $_POST['form_fecha'] . ' 00:01:00" -hlocalhost oralx_log.000001 >' . $wwwroot . 'sqlparcial.sql\' > ' . $wwwroot . 'comando');
                    exec('chmod 777 ' . $wwwroot . 'comando');
                    exec('sudo ' . $wwwroot . 'comando');
                } else { //para windows
                    exec($mysqlPath . 'mysqlbinlog -t --read-from-remote-server --database=oralxfinal -u' . $bduser . ' ' . $itp . ' --start-datetime="' . $_POST['form_fecha'] . ' 00:01:00" -hlocalhost oralx_log.000001 >' . $wwwroot . 'sqlparcial.sql\' > ' . $wwwroot . 'sqlparcial.sql');
                }
                //exec($mysqlPath.'mysqlbinlog --start-datetime="'.$_POST['form_fecha'].' 00:01:00" '.$wwwroot.'oralx_log.000001 >'.$wwwroot.'sqlparcial.sql');
                header('Content-Type: application/sql'); //Outputting the file as a csv file
                header('Content-Disposition: attachment; filename=BACKUP_PARCIAL_' . $_POST['form_fecha'] . '_' . date("Y-m-d") . '.sql');
                header('Pragma: no-cache');
                echo readfile($wwwroot . 'sqlparcial.sql');
                /* Modificado por Harvey 20111123
                  exec($mysqlPath.'mysqldump -u '.$bduser.' '.$itp.' oralxfinal auxiliares clinicas doctores especialidades visitas >'.$wwwroot.'sqlparcial.sql');
                  header('Content-Type: application/sql'); //Outputting the file as a csv file
                  header('Content-Disposition: attachment; filename=BACKUP_PARCIAL_'.date("Y-m-d").'.sql');
                  header('Pragma: no-cache');
                  echo readfile($wwwroot.'sqlparcial.sql');
                 */
            }
            // 4 = guarda el backup PARCIAL y lo restaura en el servidor
            if ($_GET['action'] == 4) {
                // guarda una copia de la tabla de exÃƒÂ¡menes.
                exec($mysqlPath . 'mysqldump -u ' . $bduser . ' ' . $itp . ' oralxfinal examenes >' . $wwwroot . 'tmpexamenes.sql');
                // realiza la restauraciÃƒÂ³n desde el archivo subido
                $target_path = $wwwroot . "uploads/";
                $target_path = $target_path . "uploadedP.sql";
                if (move_uploaded_file($_FILES['uploadedfile']['tmp_name'], $target_path)) {
                    echo "El archivo " . basename($_FILES['uploadedfile']['name']) . " ha sido importado.<br/>";
                    // borra la bd
                    //echo exec($mysqlPath.'mysql -u '.$bduser.' '.$itp.' oralxfinal -e "DROP DATABASE IF EXISTS oralxfinal"');
                    // crea la bd CREATE DATABASE IF NOT EXISTS oralxfinal
                    //echo exec($mysqlPath.'mysql -u '.$bduser.' '.$itp.' oralxfinal -e "CREATE DATABASE IF NOT EXISTS oralxfinal"');
                    //echo "La base de datos se ha vaciado<br/>";
                    //exec($mysqlPath.'mysql -u '.$bduser.' '.$itp.' oralxfinal < '.$wwwroot.'/uploads/uploadedP.sql');
                    exec($mysqlPath . 'mysql -u ' . $bduser . ' ' . $itp . ' oralxfinal < ' . $wwwroot . '/uploads/uploadedP.sql');
                    echo "La base de datos se ha restaurado correctamente desde el archivo.<br/>";
                    $this->render('backup', array());
                } else {
                    echo "Hubo un error al subir el archivo!";
                }
                // vacía la tabla exÃƒÂ¡menes
                exec($mysqlPath . 'mysql -u ' . $bduser . ' ' . $itp . ' -e "TRUNCATE TABLE examenes" oralxfinal');
                // restaura la tabla exÃƒÂ¡menes desde el archivo temporal
                exec($mysqlPath . 'mysql -u ' . $bduser . ' ' . $itp . ' oralxfinal < ' . $wwwroot . 'tmpexamenes.sql');
            }
        }
        else
            $this->render('backup', array());
    }

    public function actionView() {
        $this->render('view', array(
            'model' => $this->loadModel(),
        ));
    }

    public function actionCreate() {
        $model = new Termotablero;
        $modelArchivo = new Archivos;
        $this->performAjaxValidation($model);
        $continuar = 1;
        if (isset($_POST['Termotablero'])) {
            $model->attributes = $_POST['Termotablero'];
            $modelArchivo->nombre = $model->Path;
            $modelArchivo->save(false);
            $model->Path = "" . $modelArchivo->id;
            if ($model->save()) {
                crearEvento('TERMOTABLEROS', 'CREAR', 'Medición creada');

                $result2 = 1;
                $result1 = arreglarUltimo($model);
                if ($model->Estado < 3)
                    $result2 = crearAviso($model);
                if (($result1 == 1) && ($result2 == 1))
                    $this->redirect(array('view', 'id' => $model->id));
                else {
                    $tablero = Tableros::model()->findBySql('select * from tableros where TAG="' . $model->TAG . '"');
                    if (isset($tablero))
                        echo '<a href=/index.php/tableros/update?id=' . $tablero->id . '">Haga click aquí para ingresar el plan de mantenimiento del tablero ' . $tablero->Tablero . '</a>';
                    $continuar = 1;
                }
            }
        }
        // comprueba si el tablero tiene plan de mantenimiento.
        if (isset($_GET['id'])) {
            $tablero = Tableros::model()->findBySql('select * from tableros where TAG="' . $_GET['id'] . '"');
            if (isset($tablero)) {
                if (!isset($tablero->plan_mant_termografia)) {
                    // se setea el plan_mant_aislamiento según la nomenclatura:
                    // vib=+11000000,ais=12000000,lub=13000000,ultra=14000000,termo=15000000,termotablero=21000000
                    $tablero->plan_mant_termografia=$tablero->id+21000000;
                    // se guarda el modelo
                    $tablero->save();
                    $continuar = 1;
                }
                if ($tablero->plan_mant_termografia == 0) {
                    // se setea el plan_mant_aislamiento según la nomenclatura:
                    // vib=+11000000,ais=12000000,lub=13000000,ultra=14000000,termo=15000000,termotablero=21000000
                    $tablero->plan_mant_termografia=$tablero->id+21000000;
                    // se guarda el modelo
                    $tablero->save();
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
            $modelArchivo = Archivos::model()->findbysql("select * from archivos where id=" . $model->Path);
        else
            $modelArchivo = new Archivos();
        $this->performAjaxValidation($model);

        if (isset($_POST['Termotablero'])) {
            $model->attributes = $_POST['Termotablero'];
            $modelArchivo->nombre = $model->Path;
            $modelArchivo->save(false);
            $model->Path = "" . $modelArchivo->id;
            if ($model->save()) {

                crearEvento('TERMOTABLEROS', 'ACTUALIZAR', 'Medición actualizada');
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
            crearEvento('TERMOTABLEROS', 'BORRAR', 'Medición borrada');
            if (!isset($_GET['ajax']))
                $this->redirect(array('index'));
        }
        else
            throw new CHttpException(400,
                    Yii::t('app', 'Invalid request. Please do not repeat this request again.'));
    }

    public function actionIndex() {
        $dataProvider = new CActiveDataProvider('Termotablero');
        $this->render('index', array(
            'dataProvider' => $dataProvider,
        ));
    }

    public function actionAdmin() {
        $model = new Termotablero('search');
        if (isset($_GET['Termotablero']))
            $model->attributes = $_GET['Termotablero'];

        $this->render('admin', array(
            'model' => $model,
        ));
    }

    public function loadModel() {
        if ($this->_model === null) {
            if (isset($_GET['id']))
                $this->_model = Termotablero::model()->findbyPk($_GET['id']);
            if ($this->_model === null)
                throw new CHttpException(404, Yii::t('app', 'The requested page does not exist.'));
        }
        return $this->_model;
    }

    protected function performAjaxValidation($model) {
        if (isset($_POST['ajax']) && $_POST['ajax'] === 'termotablero-form') {
            echo CActiveForm::validate($model);
            Yii::app()->end();
        }
    }

    public function actionDynamicTableros() {
        $query = 'WHERE Area="' . $_GET['area'];
        $data = CHtml::listData(Tableros::model()->findAllbySql(
                                'SELECT TAG,concat(TAG, " - ", Tablero) as Tablero FROM tableros ' . $query . '" ORDER BY Tablero'), 'TAG', 'Tablero'
        );
        //en value 1 Areas que pertenecen al proceso.
        $value1 = "";
        $value1 = $value1 . CHtml::tag('option', array('value' => ""), CHtml::encode('Por favor seleccione el tablero'), true);
        foreach ($data as $value => $name) {
            $value1 = $value1 . CHtml::tag('option', array('value' => $value), CHtml::encode($name), true);
        }
        $response = array('value1' => $value1);
        echo CJSON::encode($response);
    }

    public function actionDynamicTableros_create() {
        $data = array();
        $data['TAG'] = $_GET['miArea'];
        echo $this->renderPartial('_adminDropdownTAG_create', $data, false, true);
    }

    public function actionDynamicTAG() {
        $data = array();
        $data["TAG"] = $_GET['TAG'];
        echo $this->renderPartial('_adminGridView', $data, false, true);
    }

    public function actionDynamicGridTAG() {
        $data = array();
        $data["TAG"] = $_GET['TAG'];
        echo $this->renderPartial('_adminGridView', $data, false, true);
    }

}

