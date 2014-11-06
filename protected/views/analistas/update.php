<?php
$this->breadcrumbs=array(
	'Analistas'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Analistas', 'url'=>array('index')),
	array('label'=>'Nuevo Analista', 'url'=>array('create')),
	array('label'=>'Detalles de Analistas', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Analistas', 'url'=>array('admin')),
);
?>
<?php
$nombre="";
$modelTMP=Motores::model()->findByAttributes(array('TAG'=>$model->TAG));
if (isset($modelTMP->Motor))
        $nombre=$modelTMP->Motor;
if (isset($modelTMP->TAG))
        $nombre=$nombre.' ('.$modelTMP->TAG.')';

?>


<?php $this->setPageTitle (' Actualizar Analistas::<?php echo $nombre?>'); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'analistas-form',
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
