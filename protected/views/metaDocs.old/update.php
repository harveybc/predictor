<?php
$this->breadcrumbs=array(
	'documentos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de documentos', 'url'=>array('index')),
	array('label'=>'Nuevo Metadocumento', 'url'=>array('create')),
	array('label'=>'Detalles Metadocumento', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar documentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle(' Actualizar Metadatos: <?php echo $model->titulo; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'meta-docs-form',
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
