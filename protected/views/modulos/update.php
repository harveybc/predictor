<?php
$this->breadcrumbs=array(
	'Módulos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Módulos', 'url'=>array('index')),
	array('label'=>'Nuevo Módulo', 'url'=>array('create')),
	array('label'=>'Detalles Módulo', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Módulos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Módulos #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'modulos-form',
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
