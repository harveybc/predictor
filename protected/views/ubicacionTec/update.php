<?php
$this->breadcrumbs=array(
	'Ubicación Técnica'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Ubicación Técnica', 'url'=>array('index')),
	array('label'=>'Nueva Ubicación Técnica', 'url'=>array('create')),
	array('label'=>'Detalles Ubicación Técnica', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Ubicación Técnica', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Ubicación Técnica #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'ubicacion-tec-form',
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
