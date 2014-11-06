<?php
$this->breadcrumbs=array(
	'Avisos Zis'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Update'),
);

$this->menu=array(
	array('label'=>'Lista de Avisos ZI', 'url'=>array('index')),
	array('label'=>'Crear Aviso ZI', 'url'=>array('create')),
	array('label'=>'Detalle de Aviso ZI', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar AvisosZI', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Avisos ZI #<?php echo $model->plan_mant; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'avisos-zi-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Update')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
