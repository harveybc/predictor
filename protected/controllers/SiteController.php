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

// recibe el modelo del motor y busca el último estado de avisosZI para vibraciones
function estadoVibraciones($modelIn) {
    $estado = 0;
    $modelTMP = new AvisosZI;
    $modelTMP = AvisosZI::model()->findBySql('select Estado from avisosZI where 
                        plan_mant="' . $modelIn->plan_mant_vibraciones . '" AND Estado>0 AND arreglado=0 order by Fecha Desc');
    if (isset($modelTMP))
        $estado = $modelTMP->Estado;
    return $estado;
}

// recibe el modelo del motor y busca el último estado de avisosZI para aislamiento
// OJO: es inverso el estado: 0=malo, 2=bueno
function estadoAislamiento($modelIn) {
    $estado = 0;
    $modelTMP = new AvisosZI;
    $modelTMP = AvisosZI::model()->findBySql('select Estado from avisosZI where 
                        plan_mant="' . $modelIn->plan_mant_aislamiento . '" AND Estado>0 AND arreglado=0 order by Fecha Desc');
    if (isset($modelTMP))
        $estado = $modelTMP->Estado;
    return $estado;
}

// recibe el modelo del motor y busca el último estado de avisosZI para aceitesn1
function estadoAceitesN1($modelIn) {
    $estado = 0;
    $modelTMP = new AvisosZI;
    $modelTMP = AvisosZI::model()->findBySql('select Estado from avisosZI where 
                        plan_mant="' . $modelIn->plan_mant_lubricantes . '" AND Estado>0 AND arreglado=0 order by Fecha Desc');
    if (isset($modelTMP))
        $estado = $modelTMP->Estado;
    return $estado;
}

// recibe el modelo del motor y busca el último estado de avisosZI para termografia
function estadoTermografia($modelIn) {
    $estado = 0;
    $modelTMP = new AvisosZI;
    $modelTMP = AvisosZI::model()->findBySql('select Estado from avisosZI where 
                        plan_mant="' . $modelIn->plan_mant_termografia . '" AND Estado>0 AND arreglado=0 order by Fecha Desc');
    if (isset($modelTMP))
        $estado = $modelTMP->Estado;
    return $estado;
}

// recibe el modelo del motor y busca el último estado de avisosZI para ultrasonido
function estadoUltrasonido($modelIn) {
    $estado = 0;
    $modelTMP = new AvisosZI;
    $modelTMP = AvisosZI::model()->findBySql('select Estado from avisosZI where 
                        plan_mant="' . $modelIn->plan_mant_ultrasonido . '" AND Estado>0 AND arreglado=0 order by Fecha Desc');
    if (isset($modelTMP))
        $estado = $modelTMP->Estado;
    return $estado;
}

function estadoTermotablero($modelIn) {
    $estado = 0;
    $modelTMP = new AvisosZI;
    $modelTMP = AvisosZI::model()->findBySql('select Estado from avisosZI where 
                        plan_mant="' . $modelIn->plan_mant_termografia . '" AND Estado>0 AND arreglado=0 order by Fecha Desc');
    if (isset($modelTMP))
        $estado = $modelTMP->Estado;
    return $estado;
}

// Dependiendo del estado, retorna el estilo del botón de motores
function colorBotonMotor($estadoIn) {
    $cadSalida = '';
    if ($estadoIn == 0)
        $cadSalida = 'cMotor';
    if ($estadoIn == 1)
        $cadSalida = 'cMotorAmarillo';
    if ($estadoIn == 2)
        $cadSalida = 'cMotorRojo';
    if ($estadoIn >= 3)
        $cadSalida = 'cMotorRojo';
    return($cadSalida);
}

// Dependiendo del estado, retorna el estilo del botón de motores
function colorBotonEquipo($estadoIn) {
    $cadSalida = '';
    if ($estadoIn == 0)
        $cadSalida = 'cEquipo';
    if ($estadoIn == 1)
        $cadSalida = 'cEquipoAmarillo';
    if ($estadoIn == 2)
        $cadSalida = 'cEquipoRojo';
    if ($estadoIn >= 3)
        $cadSalida = 'cEquipoRojo';
    return($cadSalida);
}

function colorBotonTablero($estadoIn) {
    $cadSalida = '';
    if ($estadoIn == 0)
        $cadSalida = 'cTablero';
    if ($estadoIn == 1)
        $cadSalida = 'cTableroAmarillo';
    if ($estadoIn == 2)
        $cadSalida = 'cTableroRojo';
    if ($estadoIn >= 3)
        $cadSalida = 'cTableroRojo';
    return($cadSalida);
}

// Busca el estado más grave (mayor) y retorna la imagen del semáforo
function colorEstadoMotor($eVibraciones, $eAislamiento, $eLubricantes, $eTermografia) {
    $cadSalida = '';
    $EstadoIn = 0;
    if ($EstadoIn < $eVibraciones)
        $EstadoIn = $eVibraciones;
    if ($EstadoIn < $eAislamiento)
        $EstadoIn = $eAislamiento;
    if ($EstadoIn < $eLubricantes)
        $EstadoIn = $eLubricantes;
    if ($EstadoIn < $eTermografia)
        $EstadoIn = $eTermografia;
    if ($EstadoIn == 0)
        $cadSalida = '';
    if ($EstadoIn == 1)
        $cadSalida = '<img src="/images/amarillo.gif" height="10" width="10"  class="graf_estado"/>';
    if ($EstadoIn == 2)
        $cadSalida = '<img src="/images/rojo.gif" height="10" width="10"  class="graf_estado"/>';
    if ($EstadoIn > 3)
        $cadSalida = '<img src="/images/rojo.gif" height="10" width="10"  class="graf_estado"/>';
    return($cadSalida);
}

//dibuja el semáforo para los Equipos
function colorEstadoEquipos($modelIn) {
    $cadSalida = '';
    $mayor = 0;
    $modelsTMP = AvisosZI::model()->findAllBySql('select Estado from avisosZI where 
                        Codigo="' . $modelIn->Codigo . '"
                     AND
                        Estado>0
                     AND
                        arreglado=0
                order by Fecha Desc limit 100');
    if (isset($modelsTMP)) {
        foreach ($modelsTMP as $modelTMP) {
            if (isset($modelTMP))
                if ($mayor < $modelTMP->Estado)
                    $mayor = $modelTMP->Estado;
        }
    }

// FALTA: verificar los motores y hacer or entre los arreglados de cada uno y el arreglado del equipo
    if (isset($modelTMP)) {
        $EstadoIn = $mayor;
        if ($EstadoIn == 0)
            $cadSalida = '';
        if ($EstadoIn == 1)
            $cadSalida = '<img src="/images/amarillo.gif" height="10" width="10"  class="graf_estado"/>';
        if ($EstadoIn == 2)
            $cadSalida = '<img src="/images/rojo.gif" height="10" width="10"  class="graf_estado"/>';
        if ($EstadoIn > 3)
            $cadSalida = '<img src="/images/rojo.gif" height="10" width="10"  class="graf_estado"/>';
    }
    return($cadSalida);
}

//dibuja el semáforo para los Tableros
function colorEstadoTableros($modelIn) {
    $cadSalida = '';
    $mayor = 0;
    $modelTMP = AvisosZI::model()->findBySql('select Estado from avisosZI where 
                        Codigo="' . $modelIn->id . '"
                     AND
                        Estado>0
                     AND
                        arreglado=0
                order by Fecha Desc');
    if (isset($modelTMP)) {
        if ($mayor < $modelTMP->Estado)
            $mayor = $modelTMP->Estado;
        $EstadoIn = $mayor;
        if ($EstadoIn == 0)
            $cadSalida = '';
        if ($EstadoIn == 1)
            $cadSalida = '<img src="/images/amarillo.gif" height="10" width="10" class="graf_estado" />';
        if ($EstadoIn == 2)
            $cadSalida = '<img src="/images/rojo.gif" height="10" width="10" class="graf_estado"/>';
        if ($EstadoIn > 3)
            $cadSalida = '<img src="/images/rojo.gif" height="10" width="10" class="graf_estado"/>';
    }
    return($cadSalida);
}

// CampoEquipo() Esta función Imprime el enlace y las opciones para cada Área, el
// Parámetros:
// cEquipo = Nombre del equipo
function h_encode2($entrada) {
    $salida = "H__" . $entrada;
    $salida = str_replace(" ", "-__", $salida);
    $salida = str_replace("+", "-ZZ", $salida);
    $salida = str_replace("Á", "-AA", $salida);
    $salida = str_replace("É", "-EE", $salida);
    $salida = str_replace("Í", "-II", $salida);
    $salida = str_replace("Ó", "-OO", $salida);
    $salida = str_replace("Ú", "-UU", $salida);
    $salida = str_replace("Ñ", "-NN", $salida);
    return($salida);
}

function h_decode2($entrada) {
    $salida = str_replace("H__", "", $entrada);
    $salida = str_replace("-__", " ", $salida);
    $salida = str_replace("-ZZ", "+", $salida);
    $salida = str_replace("-AA", "Á", $salida);
    $salida = str_replace("-EE", "É", $salida);
    $salida = str_replace("-II", "Í", $salida);
    $salida = str_replace("-OO", "Ó", $salida);
    $salida = str_replace("-UU", "Ú", $salida);
    $salida = str_replace("-NN", "Ñ", $salida);
    return($salida);
}

function CampoEquipo($cEquipo) {
    $cadSalida = "";
    $eUltrasonido = 0;
    $cadSalida = $cadSalida
            . CHtml::link($cEquipo->Codigo, "#", array('class' => 'cEquipoCodigo',
                'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode2($cEquipo->Equipo) . '");',
            ))
            . CHtml::link($cEquipo->Equipo, "#", array(
                'class' => 'cEquipoEspacio',
                'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode2($cEquipo->Equipo) . '");',
            ))
            . CHtml::link(CHTML::image('/images/book_open.png', 'Documentación', array('style' => 'height=10px !important;')), 'http://cbm:82/index.php/site/inicio?query=' . urlencode($cEquipo->Codigo), array(
                'onclick' => 'location.href="http://cbm:82/index.php/site/inicio?query=' . urlencode($cEquipo->Codigo) . '"',
                'class' => 'cEquipoManual'
            ))
            . CHtml::link('Detalles', '/index.php/estructura/' . $cEquipo->id, array(
                'onclick' => 'location.href="/index.php/estructura/' . $cEquipo->id . '"',
                'class' => 'cEquipo'
            ))
         . CHtml::link('Motores', '/index.php/motores/admin?id=' . $cEquipo->id, array(
                'onclick' => 'location.href="/index.php/motores/admin?id=' . $cEquipo->id . '"',
                'class' => 'cEquipo'
            ))
            . CHtml::link('Ultrasonido', '/index.php/reportes/admin?id=' . $cEquipo->id, array(
                'onclick' => 'location.href="/index.php/reportes/admin?id=' . $cEquipo->id . '"',
                'class' => colorBotonEquipo($eUltrasonido = estadoUltrasonido($cEquipo))
            ))
           . colorEstadoEquipos($cEquipo)
    ;

    // Adiciona línea horizontal de abajo
//    $cadSalida=$cadSalida.$boton = CHtml::linkButton('Ver', array('submit' => '/index.php/motores/admin'));
    return($cadSalida);
}

// CampoTablero() Esta función Imprime el enlace y las opciones para cada Área, el
// Parámetros:
// cTablero = Nombre del equipo
function CampoTablero($cTablero) {
    $cadSalida = "";
    $eTermotablero = 0;
    $cadSalida = $cadSalida

            // .CHtml::link($cTablero->TAG,"#",array("style"=>"width=20%;background-color:#FCF4A1"))
            . CHtml::link($cTablero->TAG, "#", array(
                //   'submit' => '/index.php/tableros/'.$cTablero->id,E5ACA3
                'class' => 'cTableroCodigo',
                'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode2($cTablero->TAG) . '");',
            ))
            . CHtml::link($cTablero->Tablero, "#", array(
                'class' => 'cTableroEspacio',
                'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode2($cTablero->TAG) . '");',
                    // 'submit' => '/index.php/tableros/admin?id='.$cTablero->id,
            ))

            //BORRA // .CHtml::link($cTablero->TAG,"#",array("style"=>"width=30%;background-color:#C7DAEA"))
            // .CHtml::link($cTablero->Tablero,"#",array("style"=>"width:74%;"))   
            . CHtml::link(CHTML::image('/images/book_open.png', 'Documentación', array('style' => 'height=10px !important;')), 'http://cbm:82/index.php/site/inicio?query=' . urlencode($cTablero->TAG), array(
                'onclick' => 'location.href="http://cbm:82/index.php/site/inicio?query=' . urlencode($cTablero->TAG) . '"',
                'class' => 'cEquipoManual'
            ))
            . CHtml::link('Detalles', '/index.php/tableros/' . $cTablero->id, array(
                'onclick' => 'location.href="/index.php/tableros/' . $cTablero->id . '"',
                'class' => 'cTablero'
            ))
            . CHtml::link('Termografía', '/index.php/termotablero/admin?id=' . $cTablero->id, array(
                'onclick' => 'location.href="/index.php/termotablero/admin?id=' . $cTablero->id . '"',
                'class' => colorBotonTablero($eTermotablero = estadoTermotablero($cTablero))
            ))
            . colorEstadoTableros($cTablero)

    ;


    // Adiciona línea horizontal de abajo
    //$cadSalida=$cadSalida.$boton = CHtml::linkButton('Ver', array('submit' => '/index.php/motores/admin'));
    return($cadSalida);
}

// CampoMotor() Esta función Imprime el enlace y las opciones para cada Área, el
// Parámetros:
// cMotor = Nombre del Motor
function CampoMotor($cMotor) {
    $cadSalida = "";
    // estados para determinar los colores de los botones
    $eVibraciones = 0;
    $eAislamiento = 0;
    $eAceitesN1 = 0;
    $eTermografia = 0;
    $cadSalida = $cadSalida
            . CHtml::link($cMotor->TAG, "#", array(
                //  'submit' => '/index.php/motores/'.$cMotor->id,
                'class' => 'cMotorCodigo',
                'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode2($cMotor->TAG) . '");',
            ))

            //        .CHtml::link($cMotor->TAG,array(
            //             'submit' => '/motores.php/TAG/'.$cMotor->id,
            //             'style'=>'background-color:#FCF4A1;opacity: 0.70 ;background-image:none !important;width:200px;border-left:2px solid #ffffff;border-right:2px solid #ffffff;'
            //      ))
            //      
            . CHtml::link($cMotor->Motor, '#', array(
                // 'submit' => '/index.php/motores/admin?id='.$cMotor->id,
                'class' => 'cMotorEspacio',
                'onclick' => '$("#miTree2").jstree("toggle_node", "#' . h_encode2($cMotor->TAG) . '");',
            ))
            . CHtml::link(CHTML::image('/images/book_open.png', 'Documentación', array('style' => 'height=10px !important;')), 'http://cbm:82/index.php/site/inicio?query=' . urlencode($cMotor->TAG), array(
                'onclick' => 'location.href="http://cbm:82/index.php/site/inicio?query=' . urlencode($cMotor->TAG) . '"',
                'class' => 'cEquipoManual'
            ))
            
            . CHtml::link('Detalles', '/index.php/motores/' . $cMotor->id, array(
                'onclick' => 'location.href="/index.php/motores/' . $cMotor->id . '"',
                'class' => 'cMotor'
            ))
            . CHtml::link('Aislam', '/index.php/aislamiento_tierra/admin?id=' . $cMotor->id, array(
                'onclick' => 'location.href="/index.php/aislamiento_tierra/admin?id=' . $cMotor->id . '"',
                'class' => colorBotonMotor($eAislamiento = estadoAislamiento($cMotor))
            ))
            . CHtml::link('Vibr&T°', '/index.php/vibraciones/admin?id=' . $cMotor->id, array(
                'onclick' => 'location.href="/index.php/vibraciones/admin?id=' . $cMotor->id . '"',
                'class' => colorBotonMotor($eVibraciones = estadoVibraciones($cMotor))
            ))
            . CHtml::link('Termogr', '/index.php/termomotores/admin?id=' . $cMotor->id, array(
                'onclick' => 'location.href="/index.php/termomotores/admin?id=' . $cMotor->id . '"',
                'class' => colorBotonMotor($eTermografia = estadoTermografia($cMotor))
            ))
            . CHtml::link('Lubric', '/index.php/aceitesnivel1/admin?id=' . $cMotor->id, array(
                'onclick' => 'location.href="/index.php/aceitesnivel1/admin?id=' . $cMotor->id . '"',
                'class' => colorBotonMotor($eAceitesN1 = estadoAceitesN1($cMotor))
            ))
            . colorEstadoMotor($eVibraciones, $eAislamiento, $eAceitesN1, $eTermografia);

    // Adiciona línea horizontal de abajo
    //$cadSalida=$cadSalida.$boton = CHtml::linkButton('Ver', array('submit' => '/index.php/motores/admin'));
    return($cadSalida);
}


class SiteController extends Controller {

    public $layout = '//layouts/responsiveLayout';
    public $defaultAction = 'inicio';

    /**
     * Declares class-based actions.
     */
    public function actions() {
        return array(
            // captcha action renders the CAPTCHA image displayed on the contact page
            'captcha' => array(
                'class' => 'CCaptchaAction',
                'backColor' => 0xFFFFFF,
            ),
            // page action renders "static" pages stored under 'protected/views/site/pages'
            // They can be accessed via: index.php?r=site/page&view=FileName
            'page' => array(
                'class' => 'CViewAction',
            ),
            'sitemap' => array(
                'class' => 'ext.sitemap.ESitemapAction',
                'importListMethod' => 'getBaseSitePageList',
                'classConfig' => array(
                    array('baseModel' => 'reportes',
                        'route' => '/reportes/index',
                    //  'params'=>array('id'=>'taskId')),         
                    ),
                ),
            ),
            'sitemapxml' => array(
                'class' => 'ext.sitemap.ESitemapXMLAction',
                'classConfig' => array(
                    array('baseModel' => 'Task',
                        'route' => '/task/view',
                        'params' => array('id' => 'taskId')
                    ),
                ),
                //'bypassLogs'=>true, // if using yii debug toolbar enable this line
                'importListMethod' => 'getBaseSitePageList',
            ),
        );
    }

    public function filters() {
        return array(
            'accessControl',
        );
    }

    public function accessRules() {
        return array(
            array('allow',
                'actions' => array('Login', 'Contact', 'sitemap', 'error', 'dashboard'),
                'users' => array('*'),
            ),
            array('allow',
                'actions' => array('inicio', 'index', 'logout', 'dynamicResumen', 'getArbol', 'page', 'buscar'),
                'users' => array('@'),
            ),
            array('deny',
                'users' => array('*'),
            ),
        );
    }

    public function actionBuscar() {
        // renders the view file 'protected/views/site/index.php'
        // using the default layout 'protected/views/layouts/main.php'
        $this->render('buscar');
    }

    /**
     * This is the default 'index' action that is invoked
     * when an action is not explicitly requested by users.
     */
    public function actionIndex() {
        // renders the view file 'protected/views/site/index.php'
        // using the default layout 'protected/views/layouts/main.php'
        $this->render('index');
    }

    /**
     * This is the action to handle external exceptions.
     */
    public function actionError() {
        if ($error = Yii::app()->errorHandler->error) {
            if (Yii::app()->request->isAjaxRequest)
                echo $error['message'];
            else
                $this->render('error', $error);
        }
    }

    /**
     * Displays the contact page
     */
    public function actionContact() {
        $model = new ContactForm;
        if (isset($_POST['ContactForm'])) {
            $model->attributes = $_POST['ContactForm'];
            if ($model->validate()) {
                $headers = "From: {$model->email}\r\nReply-To: {$model->email}";
                mail(Yii::app()->params['adminEmail'], $model->subject, $model->body, $headers);
                Yii::app()->user->setFlash('contact', 'Thank you for contacting us. We will respond to you as soon as possible.');
                $this->refresh();
            }
        }
        $this->render('contact', array('model' => $model));
    }

    /**
     * Displays the login page
     */
    public function actionLogin() {
        $model = new LoginForm;

        // if it is ajax validation request
        if (isset($_POST['ajax']) && $_POST['ajax'] === 'login-form') {
            echo CActiveForm::validate($model);
            Yii::app()->end();
        }

        // collect user input data
        if (isset($_POST['LoginForm'])) {
            $model->attributes = $_POST['LoginForm'];
            // validate user input and redirect to the previous page if valid
            if ($model->validate() && $model->login()) {
                crearEvento('SITE', 'LOGIN_OK', 'Usuario autenticado');
                $this->redirect(Yii::app()->user->returnUrl);
            }
        }
        // display the login form
        $this->render('login', array('model' => $model));
    }

    /**
     * Logs out the current user and redirect to homepage.
     */
    public function actionLogout() {
        Yii::app()->user->logout();
        $this->redirect(Yii::app()->homeUrl);
    }

    public function actionInicio() {
        // renders the view file 'protected/views/site/index.php'
        // using the default layout 'protected/views/layouts/main.php'
        
        $this->render('inicio');
    }
    public function actionDashboard() {
        // renders the view file 'protected/views/site/index.php'
        // using the default layout 'protected/views/layouts/main.php'
        
        $this->render('dashboard');
    }
    public function actionDynamicResumen() {
        $data = array();
        $data["proceso"] = $_GET['proceso'];
        echo $this->renderPartial('/site/_GridResumen', $data, false, true);
    }

    public function actionGetArbol() {
        $dataTree = "";
        if ($_GET['tipo'] == "Tablero") {
            //muestra los tableros del id Proceso en la tabla tableros
            $tmp = h_decode2($_GET['id']);
            $tmp = str_replace("_tableros", "", $tmp);
            $tableros = Tableros::model()->findAllBySql('SELECT * FROM tableros WHERE Area="' . $tmp . '" ORDER BY Tablero');
            foreach ($tableros as $tablero) {
                $dataTree = $dataTree . '<li id="' . h_encode2($tablero->TAG) . '">'; // Tablero
                $dataTree = $dataTree . CampoTablero($tablero);
                $dataTree = $dataTree . "</li>"; // Tablero
            }
        }
        if ($_GET['tipo'] == "Equipos") {
            //muestra los equipos del id Proceso en la tabla estructura
            $tmp = h_decode2($_GET['id']);
            $tmp = str_replace("_equipos", "", $tmp);
            $equipos = Estructura::model()->findAllBySql('SELECT * FROM estructura WHERE Area="' . $tmp . '" ORDER BY Equipo');
            // para cada proceso
            foreach ($equipos as $equipo) {
                $consultaSQL = 'SELECT COUNT(*) AS contador FROM motores WHERE (Equipo ="' . $equipo->Equipo . '") LIMIT 1';
                $command = Yii::app()->db->createCommand($consultaSQL);
                $resultados = $command->queryAll();
                $tipoE = (0 + $resultados[0]['contador']) > 0 ? "jstree-closed" : "";
                $dataTree = $dataTree . '<li id="' . h_encode2($equipo->Equipo) . '" tipo="Equipo" class="' . $tipoE . '">'; // Equipo
                $dataTree = $dataTree . CampoEquipo($equipo);

               // $dataTree = $dataTree . "</ul>"; // Equipo
                $dataTree = $dataTree . "</li>"; // Equipo
            }
        }
        if ($_GET['tipo'] == "Equipo") {
            //muestra motores del id Equipo en la tabla motores
            $motores = Motores::model()->findAllBySql('SELECT * FROM motores WHERE Equipo="' . h_decode2($_GET['id']) . '" ORDER BY Motor');
            foreach ($motores as $motor) {
                $dataTree = $dataTree . '<li id="' . h_encode2($motor->TAG) . '">'; // Motor
                $dataTree = $dataTree . CampoMotor($motor);
                $dataTree = $dataTree . "</li>"; // Motor
            }
        }
        echo $dataTree;
    }

}

