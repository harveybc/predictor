<?php
$this->breadcrumbs=array(
	'Tableros'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Tableros', 'url'=>array('index')),
	array('label'=>'Nuevo Tablero', 'url'=>array('create')),
	array('label'=>'Ver Tableros', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Tableros', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=Tableros::model()->findByAttributes(array('TAG'=>$model->TAG));
if (isset($modelTMP->Tablero))
        $nombre=$modelTMP->Tablero;
if (isset($modelTMP->TAG))
        $nombre=$nombre.' ('.$modelTMP->TAG.')';

?>
<?php $this->setPageTitle (' Actualizar Tablero:<?php echo $nombre?>'); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'tableros-form',
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
