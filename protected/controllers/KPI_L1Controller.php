<?php

class KPI_L1Controller extends Controller {

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
                'actions' => array('index', 'view', 'dynamicInfo', 'dynamicImage','dynamicImageGen'),
                'users' => array('*'),
            ),
            array('allow',
                'actions' => array('create', 'update'),
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
        $model = new KPI_L1;

        $this->performAjaxValidation($model);

        if (isset($_POST['KPI_L1'])) {
            $model->attributes = $_POST['KPI_L1'];


            if ($model->save())
                $this->redirect(array('view', 'id' => $model->id));
        }

        $this->render('create', array(
            'model' => $model,
        ));
    }

    public function actionUpdate() {
        $model = $this->loadModel();

        $this->performAjaxValidation($model);

        if (isset($_POST['KPI_L1'])) {
            $model->attributes = $_POST['KPI_L1'];

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
        $dataProvider = new CActiveDataProvider('KPI_L1');
        $this->render('index', array(
            'dataProvider' => $dataProvider,
        ));
    }

    public function actionAdmin() {
        $model = new KPI_L1('search');
        if (isset($_GET['KPI_L1']))
            $model->attributes = $_GET['KPI_L1'];

        $this->render('admin', array(
            'model' => $model,
        ));
    }

    public function loadModel() {
        if ($this->_model === null) {
            if (isset($_GET['id']))
                $this->_model = KPI_L1::model()->findbyPk($_GET['id']);
            if ($this->_model === null)
                throw new CHttpException(404, Yii::t('app', 'The requested page does not exist.'));
        }
        return $this->_model;
    }

    protected function performAjaxValidation($model) {
        if (isset($_POST['ajax']) && $_POST['ajax'] === 'kpi--l1-form') {
            echo CActiveForm::validate($model);
            Yii::app()->end();
        }
    }

    public function ActionDynamicInfo() {
        
        $modelo1 = KPI_L1::model()->findBySQL("select * from KPI_L1 order by ID desc ");
        $modelo2 = KPI_L2::model()->findBySQL("select * from KPI_L2 order by ID desc ");
        $eff_1=sprintf("%.2f%%",$modelo1->Eff_Shift);
        $eff_2=sprintf("%.2f%%",$modelo2->Eff_Shift);
        if ($modelo1->Eff_Shift>100)
            $eff_1="ERR";
        if ($modelo2->Eff_Shift>100)
            $eff_2="ERR";
            
        $salida = sprintf('
            <img src="/index.php/KPI_L1/dynamicImageGen?sb_graph_range='.$_POST['sb_graph_range'].'" style="width:100%%;height:90px;"/>
                      <div class="sb_data" style="color:#999;display:block;">Actual:</div> 
                      <div class="sb_data" style="color:#C44;">L1=%s</div> 
            <div class="sb_data" style="color:#48C;">L2=%s
            ', $eff_1
                , $eff_2);

        /*            ',$modelo1->Eff_Shift,$modelo1->Count_Pall_Batch,$modelo1->Count_Fill_Shift,$modelo1->Count_Fill_Batch
          ,$modelo2->Eff_Shift,$modelo2->Count_Pall_Batch,$modelo2->Count_Fill_Shift,$modelo2->Count_Fill_Batch); */
        echo $salida;
    }

    /*     * ***TEST ********* */

    public function ActionDynamicImageGen() {
        //Yii::import('mods/phpMyGraph5.0.php'); 
        Yii::import('ext.phpMyGraph5', true);
        //Set config directives 
        $cfg['title'] = 'Example graph';
        $cfg['width'] = 324;
        $cfg['height'] = 200;
        $cfg['type'] = 'gif';
$cfg['zero-line-visible'] = 0;
$cfg['average-line-visible'] = 0;
$cfg['column-divider-visible'] = 0;
$cfg['title-visible'] = 0;
$cfg['label-visible'] = 1;
$cfg['value-label-visible'] = 0;
$cfg['key-visible'] = 0;
$cfg['value-font-size'] = 1;
$cfg['value-visible'] = 1;


        // calcula divisor y número de consultas:
        $segundos=360*60; //6horas por defecto
        if (isset($_GET['sb_graph_range']))
            $segundos = 60*$_GET['sb_graph_range'];
        $num_consultas=($segundos>600)?300:100;
        $divisor=round($segundos/($num_consultas*6));

        // Set data 1 sb_graph_range
        // busca último id
         $id_last=1;
            $modeloTMP=KPI_L1::model()->findBySql("select id from KPI_L1 order by id desc limit 1");
            if (isset($modeloTMP)){
                    $id_last=$modeloTMP->id;
                }
        $modelos=  KPI_L1::model()->findAllBySql('
            select id,Eff_Shift,Fecha from KPI_L1 where MOD(('.$id_last.'-id),'.$divisor.')=0 order by id desc limit '.$num_consultas.'
            ');
        $data1 = array();
        if (isset($modelos))
            $count_h=0;
        foreach ($modelos as $modelo) 
            {
                if( (isset($modelo->Eff_Shift))&&(isset($modelo->Fecha)))
                    $data1[$count_h]=($modelo->Eff_Shift>100)?0:floor($modelo->Eff_Shift);
                $count_h++;
            }
        // Set data 1 
        // busca último id
         $id_last=1;
            $modeloTMP=KPI_L2::model()->findBySql("select id from KPI_L2 order by id desc limit 1");
            if (isset($modeloTMP)){
                    $id_last=$modeloTMP->id;
                }
        $modelos=  KPI_L2::model()->findAllBySql('
            select id,Eff_Shift,Fecha from KPI_L2 where MOD(('.$id_last.'-id),'.$divisor.')=0 order by id desc limit '.$num_consultas.'
            ');
        $data2 = array();
        if (isset($modelos))
            $count_h=0;
        foreach ($modelos as $modelo) 
            {
                if( (isset($modelo->Eff_Shift))&&(isset($modelo->Fecha)))
                    $data2[$count_h]=($modelo->Eff_Shift>100)?0:floor($modelo->Eff_Shift);
                $count_h++;
            }
        //Create phpMyGraph instance 
        $graph = new verticalLineGraph();

        //Parse 
        return $graph->parseCompare(array_reverse($data1),array_reverse($data2), $cfg);
    }
    public function ActionDynamicImage() 
            {
        //echo 'MTPHK';
        echo '<img src="/index.php/KPI_L1/dynamicImageGen" />';
        }

}
