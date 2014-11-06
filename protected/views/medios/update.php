<?php
$this->breadcrumbs=array(
	'Medios de Publicación'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Medios', 'url'=>array('index')),
	array('label'=>'Nuevo Medios', 'url'=>array('create')),
	array('label'=>'Detalles Medio', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Medios', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Medios de Publicación #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'medios-form',
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
