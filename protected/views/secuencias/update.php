<?php
$this->breadcrumbs=array(
	'Secuencias'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Secuencias', 'url'=>array('index')),
	array('label'=>'Nueva Secuencia', 'url'=>array('create')),
	array('label'=>'Detalles Secuencia', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Secuencias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Secuencias #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'secuencias-form',
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
