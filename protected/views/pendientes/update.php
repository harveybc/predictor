
<?php
$this->breadcrumbs=array(
	'Pendientes'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Update'),
);

$this->menu=array(
	array('label'=>'List Pendientes', 'url'=>array('index')),
	array('label'=>'Create Pendientes', 'url'=>array('create')),
	array('label'=>'View Pendientes', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Manage Pendientes', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Update Pendientes #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'pendientes-form',
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
