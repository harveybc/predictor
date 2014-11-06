<?php
$this->breadcrumbs=array(
	'Evaluaciones Generales'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Evaluaciones Generales', 'url'=>array('index')),
	array('label'=>'Nueva Evaluación General', 'url'=>array('create')),
	array('label'=>'Detalles Evaluación General', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Evaluaciones Generales', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Evaluaciones Generales #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'evaluaciones-generales-form',
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
