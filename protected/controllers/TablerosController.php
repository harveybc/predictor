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
class TablerosController extends Controller
{
	public $layout='//layouts/responsiveLayout';
	private $_model;

	public function filters()
	{
		return array(
			'accessControl', 
		);
	}

	public function accessRules()
	{
		return array(
			array('allow',  
				'actions'=>array('index','view','DynamicTableros','DynamicArea','DynamicTableroDropDown'),
				'users'=>array('*'),
			),
			array('allow', 
				'actions'=>array('update','admin','create'),
				'users'=>array('@'),
			),
			array('allow', 
				'actions'=>array('delete'),
				'users'=>array('admin'),
			),
			array('deny', 
				'users'=>array('*'),
			),
		);
	}

	public function actionView()
	{
		$this->render('view',array(
			'model'=>$this->loadModel(),
		));
	}

	public function actionCreate()
	{
		$model=new Tableros;

		$this->performAjaxValidation($model);

		if(isset($_POST['Tableros']))
		{
			$model->attributes=$_POST['Tableros'];
		

			if($model->save())
                        {
                                                            crearEvento('TABLEROS','CREAR','Tablero creado');
				$this->redirect(array('view','id'=>$model->id));
                        }
		}

		$this->render('create',array(
			'model'=>$model,
		));
	}

	public function actionUpdate()
	{
		$model=$this->loadModel();

		$this->performAjaxValidation($model);

		if(isset($_POST['Tableros']))
		{
			$model->attributes=$_POST['Tableros'];
		
			if($model->save())
                        {
                                crearEvento('TABLEROS','ACTUALIZAR','Tablero actualizado');

				$this->redirect('/index.php/site/inicio');
                        }
		}

		$this->render('update',array(
			'model'=>$model,
		));
	}

	public function actionDelete()
	{
		if(Yii::app()->request->isPostRequest)
		{
			$this->loadModel()->delete();
                                                            crearEvento('TABLEROS','BORRAR','Tablero borrado');

			if(!isset($_GET['ajax']))
				$this->redirect(array('index'));
		}
		else
			throw new CHttpException(400,
					Yii::t('app', 'Invalid request. Please do not repeat this request again.'));
	}

	public function actionIndex()
	{
		$dataProvider=new CActiveDataProvider('Tableros');
		$this->render('index',array(
			'dataProvider'=>$dataProvider,
		));
	}

	public function actionAdmin()
	{
		$model=new Tableros('search');
		if(isset($_GET['Tableros']))
			$model->attributes=$_GET['Tableros'];

		$this->render('admin',array(
			'model'=>$model,
		));
	}

	public function loadModel()
	{
		if($this->_model===null)
		{
			if(isset($_GET['id']))
				$this->_model=Tableros::model()->findbyPk($_GET['id']);
			if($this->_model===null)
				throw new CHttpException(404, Yii::t('app', 'The requested page does not exist.'));
		}
		return $this->_model;
	}

	protected function performAjaxValidation($model)
	{
		if(isset($_POST['ajax']) && $_POST['ajax']==='tableros-form')
		{
			echo CActiveForm::validate($model);
			Yii::app()->end();
		}
	}
         public function actionDynamicTableros() {
        $data = array();
        $data["tableros"] = $_GET['area'];
        echo $this->renderPartial('_adminGridView', $data, false, true);
        
        
    }
    
        public function actionDynamicTableroDropDown()
        {
            $query = 'WHERE Area="' . $_GET['area'];
            $data = CHtml::listData(Tableros::model()->findAllbySql(
                        'SELECT TAG, Tablero FROM tableros ' . $query . '" ORDER BY Tablero'),
                        'TAG',
                        'Tablero'
            );
            //en value 1 Areas que pertenecen al proceso.
            $value1 = "";
            $value1 = $value1 . CHtml::tag('option',
                                               array ('value' => ""),
                                               CHtml::encode('Por favor seleccione el tablero'),
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
    
            public function actionDynamicArea()
        {
            $query = 'WHERE Proceso="' . $_GET['proceso'];
            $data = CHtml::listData(Estructura::model()->findAllbySql(
                        'SELECT DISTINCT Area FROM tableros ' . $query . '" ORDER BY Area'),
                        'Area',
                        'Area'
            );
            //en value 1 Areas que pertenecen al proceso.
            $value1 = "";
            $value1 = $value1 . CHtml::tag('option',
                                               array ('value' => ""),
                                               CHtml::encode("Seleccione un proceso"),
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
}
