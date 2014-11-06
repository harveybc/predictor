<?php
$this->breadcrumbs=array(
	'Unidades Documentales'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Unidades', 'url'=>array('index')),
	array('label'=>'Nueva Unidad', 'url'=>array('create')),
	array('label'=>'Detalles Unidad', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Unidades', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle(' Actualizar Unidad Documental: <?php echo $model->descripcion; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'documentos-form',
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
