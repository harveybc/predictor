<?php
$this->breadcrumbs=array(
	'Kpi  L2s'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Update'),
);

$this->menu=array(
	array('label'=>'List KPI_L2', 'url'=>array('index')),
	array('label'=>'Create KPI_L2', 'url'=>array('create')),
	array('label'=>'View KPI_L2', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Manage KPI_L2', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Update KPI_L2 #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'kpi--l2-form',
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
