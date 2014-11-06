<?php
$this->breadcrumbs=array(
	'Aislamiento Acometidas'=>array('index'),
	$model->Toma=>array('view','id'=>$model->Toma),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Mediciones', 'url'=>array('index')),
	array('label'=>'Nueva Medición', 'url'=>array('create')),
	array('label'=>'Detalles detalles', 'url'=>array('view', 'id'=>$model->Toma)),
	array('label'=>'Gestionar Mediciones', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=Motores::model()->findByAttributes(array('TAG'=>$model->TAG));
if (isset($modelTMP->Motor))
        $nombre=$modelTMP->Motor;
if (isset($modelTMP->TAG))
        $nombre=$nombre.' ('.$modelTMP->TAG.')';

?>

<?php $this->setPageTitle (' Actualizar Medición Aislamiento Acometida de:<?php echo $nombre?>'); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'aislamiento-acometida-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
